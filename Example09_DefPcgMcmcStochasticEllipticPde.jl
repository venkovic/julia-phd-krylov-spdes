push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./MyPreconditioners/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: cg, pcg, defpcg, eigpcg, eigdefpcg
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using MyPreconditioners: BJPreconditioner, BJop,
                         Cholesky16, get_cholesky16,
                         Cholesky32, get_cholesky32

using Utils: space_println, printlnln, save_deflated_system

using NPZ: npzread, npzwrite
using Random: seed!
using LinearMaps: LinearMap
import SuiteSparse

tentative_nnode = 200_000
load_existing_mesh = false

ndom = 20
load_existing_partition = false

nbj = ndom

nvec = ndom + 5
spdim = floor(Int, 2.5 * nvec)

do_amg_0_pcg = false
do_lorasc_0_pcg = false
verbose = true

nchains = 50
nsmp = 5
seed!(481_456)

model = "SExp"
sig2 = 1.
L = .1

root_fname = get_root_filename(model, sig2, L, tentative_nnode)

if load_existing_mesh
  cells, points, point_markers, cell_neighbors = load_mesh(tentative_nnode)
else
  mesh = get_mesh(tentative_nnode)
  cells = mesh.cell
  points = mesh.point
  point_markers = mesh.point_marker
  cell_neighbors = mesh.cell_neighbor
end

dirichlet_inds_g2l, not_dirichlet_inds_g2l,
dirichlet_inds_l2g, not_dirichlet_inds_l2g =
get_dirichlet_inds(points, point_markers)

n = not_dirichlet_inds_g2l.count
nnode = mesh.n_point

function f(x::Float64, y::Float64)
  return -1.
end
  
function uexact(xx::Float64, yy::Float64)
  return .734
end

println()
space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")


#
# Assembly of constant preconditioners
#
g_0 = zeros(Float64, nnode)

printlnln("do_isotropic_elliptic_assembly for ξ = 0 ...")
A_0, _ = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(g_0), f, uexact)

printlnln("prepare amg_0 preconditioner for A_0 ...")
Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);

printlnln("prepare_lorasc_precond for A_0 ...")
Π_lorasc_0 = @time prepare_lorasc_precond(tentative_nnode,
                                          ndom,
                                          cells,
                                          points,
                                          cell_neighbors,
                                          exp.(g_0),
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          f,
                                          uexact)

printlnln("prepare bj_0 preconditioner for A_0 ...")
Π_bj_0 = @time BJPreconditioner(nbj, A_0);

printlnln("prepare chol16_0 preconditioner for A_0 ...")
Π_chol16_0 = @time get_cholesky16(A_0);

#
# Load kl representation and prepare mcmc sampler
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
                                              
function test_one_chain_01(Π_amg_0,
                           Π_lorasc_0::LorascPreconditioner,
                           Π_bj_0::BJop,
                           Π_chol16_0::Cholesky16,
                           verbose::Bool,
                           do_amg_0_pcg::Bool,
                           do_lorasc_0_pcg::Bool,
                           nsmp::Int,
                           Λ::Array{Float64,1},
                           Ψ::Array{Float64,2};
                           save_pcg_results=false)
  
  sampler = prepare_mcmc_sampler(Λ, Ψ)

  ndom, = size(Π_lorasc_0.x_Id)
  methods = ["amg_0-eigdefpcg",
             "lorasc$(ndom)_0-eigdefpcg",
             "bj$(nbj)_0-eigdefpcg",
             "chol16_0-eigdefpcg"]
  do_amg_0_pcg ? push!(methods, "amg_0-pcg") : nothing
  do_lorasc_0_pcg ? push!(methods, "lorasc$(ndom)_0-pcg") : nothing

  iter = Dict{String,Array{Int,1}}()
  for method in methods
    iter[method] = Array{Int,1}(undef, nsmp)
  end

  verbose ? println("\n1 / $nsmp") : nothing
  verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
  Δt = @elapsed A, b = do_isotropic_elliptic_assembly(cells, points,
                                                      dirichlet_inds_g2l,
                                                      not_dirichlet_inds_g2l,
                                                      point_markers,
                                                      exp.(sampler.g),
                                                      f, uexact)
  verbose ? println("$Δt seconds") : nothing

  x = zeros(Float64, A.n)

  if do_amg_0_pcg
    verbose ? print("amg_0-pcg of A * u = b ... ") : nothing
    Δt = @elapsed _, it, _  = pcg(A, b, M=Π_amg_0)
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["amg_0-pcg"][1] = it
  end
  
  if do_lorasc_0_pcg
    verbose ? print("lorasc$(ndom)_0-pcg solve of A * u = b ...") : nothing
    Δt = @elapsed _, it, _ = pcg(A, b, M=Π_lorasc_0)
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["lorasc$(ndom)_0-pcg"][1] = it
  end

  print("amg_0-eigpcg solve of A * u = b ...")
  Δt = @elapsed _, it, _, W_amg = eigpcg(A, b, x, Π_amg_0, nvec, spdim)
  verbose ? println("$Δt seconds, iter = $it") : nothing
  iter["amg_0-eigdefpcg"][1] = it

  print("lorasc_0-eigpcg solve of A * u = b ...")
  x .= 0.
  Δt = @elapsed _, it, _, W_lorasc = eigpcg(A, b, x, Π_lorasc_0, nvec, spdim)
  verbose ? println("$Δt seconds, iter = $it") : nothing
  iter["lorasc$(ndom)_0-eigdefpcg"][1] = it

  print("bj_0-eigpcg solve of A * u = b ...")
  x .= 0.
  Δt = @elapsed _, it, _, W_bj = eigpcg(A, b, x, Π_bj_0, nvec, spdim)
  verbose ? println("$Δt seconds, iter = $it") : nothing
  iter["bj$(nbj)_0-eigdefpcg"][1] = it

  print("chol16_0-eigpcg solve of A * u = b ...")
  x .= 0.
  Δt = @elapsed _, it, _, W_chol16 = eigpcg(A, b, x, Π_chol16_0, nvec, spdim)
  verbose ? println("$Δt seconds, iter = $it") : nothing
  iter["chol16_0-eigdefpcg"][1] = it

  #
  # Sample ξ by mcmc and solve linear systems by def-pcg with 
  # online eigenvectors approximation
  #
  cnt_reals = 1
  for s in 2:nsmp

    verbose ? print("\n$s / $nsmp") : nothing
    cnt_reals += draw!(sampler)
    verbose ? println(" (acceptance frequency: $(s / cnt_reals))") : nothing

    verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
    Δt = @elapsed update_isotropic_elliptic_assembly!(A, b,
                                                      cells, points,
                                                      dirichlet_inds_g2l,
                                                      not_dirichlet_inds_g2l,
                                                      point_markers,
                                                      exp.(sampler.g),
                                                      f, uexact)
    verbose ? println("$Δt seconds") : nothing

    if do_amg_0_pcg
      verbose ? print("amg_0-pcg of A * u = b ... ") : nothing
      Δt = @elapsed _, it, _ = pcg(A, b, M=Π_amg_0)
      verbose ? println("$Δt seconds, iter = $it") : nothing
      iter["amg_0-pcg"][s] = it
    end

    if do_lorasc_0_pcg
      print("lorasc$(ndom)_0-pcg solve of A * u = b ...")
      Δt = @elapsed _, it, _ = pcg(A, b, M=Π_lorasc_0)
      verbose ? println("$Δt seconds, iter = $it") : nothing
    end

    print("amg_0-eigdefpcg solve of A * u = b ...")
    x .= 0.
    try
      Δt = @elapsed _, it, _, W_amg = eigdefpcg(A, b, x, Π_amg_0, W_amg, spdim)
    catch
      x .= 0.
      Δt = @elapsed _, it, _, W_amg = eigpcg(A, b, x, Π_amg_0, nvec, spdim)
      save_deflated_system(A, b, W_amg, s, "amg", print_error=true)
    end 
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["amg_0-eigdefpcg"][s] = it

    print("lorasc$(ndom)_0-eigdefpcg solve of A * u = b ...")
    x .= 0.
    try
      Δt = @elapsed _, it, _, W_lorasc = eigdefpcg(A, b, x, Π_lorasc_0, W_lorasc, spdim)
    catch 
      x .= 0.
      Δt = @elapsed _, it, _, W_lorasc = eigpcg(A, b, x, Π_lorasc_0, nvec, spdim)
      save_deflated_system(A, b, W_lorasc, s, "lorasc", print_error=true)
    end 
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["lorasc$(ndom)_0-eigdefpcg"][s] = it

    print("bj$(nbj)_0-eigdefpcg solve of A * u = b ...")
    x .= 0.
    try
      Δt = @elapsed _, it, _, W_bj = eigdefpcg(A, b, x, Π_bj_0, W_bj, spdim)
    catch 
      x .= 0.
      Δt = @elapsed _, it, _, W_bj = eigpcg(A, b, x, Π_bj_0, nvec, spdim)
      save_deflated_system(A, b, W_bj, s, "bj", print_error=true)
    end 
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["bj$(nbj)_0-eigdefpcg"][s] = it

    print("chol16_0-eigdefpcg solve of A * u = b ...")
    x .= 0.
    try
      Δt = @elapsed _, it, _, W_chol16 = eigdefpcg(A, b, x, Π_chol16_0, W_chol16, spdim)
    catch 
      x .= 0.
      Δt = @elapsed _, it, _, W_chol16 = eigpcg(A, b, x, Π_chol16_0, nvec, spdim)
      save_deflated_system(A, b, W_chol16, s, "chol16", print_error=true)
    end 
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["chol16_0-eigdefpcg"][s] = it
  end

  if save_pcg_results
    do_amg_0_pcg ? npzwrite("data/test01_$(root_fname)_amg_0-pcg.it.npz",
                            iter["amg_0-pcg"]) : nothing
    do_lorasc_0_pcg ? npzwrite("data/test01_$(root_fname)_lorasc$(ndom)_0-pcg.it.npz",
                               iter["lorasc$(ndom)_0-pcg"]) : nothing
    npzwrite("data/test01_$(root_fname)_amg_0-eigdefpcg_nvec$(nvec)_spdim$(spdim).it.npz",
             iter["amg_0-eigdefpcg"])
    npzwrite("data/test01_$(root_fname)_lorasc$(ndom)_0-eigdefpcg_nvec$(nvec)_sdpim$(spdim).it.npz",
             iter["lorasc$(ndom)_0-eigdefpcg"])
    npzwrite("data/test01_$(root_fname)_bj$(nbj)_0-eigdefpcg_nvec$(nvec)_sdpim$(spdim).it.npz",
             iter["bj$(nbj)_0-eigdefpcg"])
    npzwrite("data/test01_$(root_fname)_chol16_0-eigdefpcg_nvec$(nvec)_sdpim$(spdim).it.npz",
             iter["chol16_0-eigdefpcg"])
  end

  return iter
end


function test_several_chains_01(nchains::Int,
                                Π_amg_0,
                                Π_lorasc_0::LorascPreconditioner,
                                Π_bj_0::BJop,
                                Π_chol16_0::Cholesky16,
                                verbose::Bool,
                                do_amg_0_pcg::Bool,
                                do_lorasc_0_pcg::Bool,
                                nsmp::Int,
                                Λ::Array{Float64,1},
                                Ψ::Array{Float64,2};
                                save_progressively=false)

  iters = Dict{String,Array{Int,2}}()

  for ichain in 1:nchains

    println("\n\nworking on chain $ichain / $nchains ...")

    iter = test_one_chain_01(Π_amg_0,
                             Π_lorasc_0,
                             Π_bj_0,
                             Π_chol16_0,
                             verbose,
                             do_amg_0_pcg,
                             do_lorasc_0_pcg,
                             nsmp,
                             Λ,
                             Ψ)
   
    for (method, _iter) in iter
      if haskey(iters, method) 
        iters[method] = hcat(iters[method], _iter)
      else
        iters[method] = reshape(_iter, length(_iter), 1)
      end
    end
                                
    npzwrite("data/test01_$(root_fname)_amg_0-eigdefpcg_nvec$(nvec)_spdim$(spdim).it.npz",
             iters["amg_0-eigdefpcg"])
    npzwrite("data/test01_$(root_fname)_lorasc$(ndom)_0-eigdefpcg_nvec$(nvec)_spdim$(spdim).it.npz",
             iters["lorasc$(ndom)_0-eigdefpcg"])
    npzwrite("data/test01_$(root_fname)_bj$(nbj)_0-eigdefpcg_nvec$(nvec)_spdim$(spdim).it.npz",
             iters["bj$(nbj)_0-eigdefpcg"])
    npzwrite("data/test01_$(root_fname)_chol16_0-eigdefpcg_nvec$(nvec)_spdim$(spdim).it.npz",
             iters["chol16_0-eigdefpcg"])

    println("\n\n ... done working on chain $ichain / $nchains.")
  end

  return iters
end


iters = test_several_chains_01(nchains,
                               Π_amg_0,
                               Π_lorasc_0,
                               Π_bj_0,
                               Π_chol16_0,
                               verbose,
                               do_amg_0_pcg,
                               do_lorasc_0_pcg,
                               nsmp,
                               Λ,
                               Ψ)