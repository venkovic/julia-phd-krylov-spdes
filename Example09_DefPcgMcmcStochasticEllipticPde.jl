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

using Utils: space_println, printlnln

using NPZ: npzread, npzwrite
using Random: seed!
using LinearMaps: LinearMap
import SuiteSparse


tentative_nnode = 50_000
load_existing_mesh = false

ndom = 40
load_existing_partition = false

nbj = ndom

nvec = ndom + 5
spdim = floor(Int, 2 * nvec)

do_lorasc_0_pcg = false
verbose = true

nsmp = 30
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
                                              
sampler = prepare_mcmc_sampler(Λ, Ψ)

function test01(Π_amg_0,
                Π_lorasc_0::LorascPreconditioner,
                Π_bj_0::BJop,
                Π_chol16_0::Cholesky16,
                sampler::McmcSampler,
                verbose::Bool,
                do_lorasc_0_pcg::Bool,
                nsmp::Int)
  
  ndom, = size(Π_lorasc_0.x_Id)
  methods = ["amg_0-pcg",
             "lorasc$(ndom)_0-eigdefpcg",
             "bj$(nbj)_0-eigdefpcg",
             "chol16_0-eigdefpcg"]

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

  verbose ? print("amg_0-pcg of A * u = b ... ") : nothing
  Δt = @elapsed _, it, _  = pcg(A, b, M=Π_amg_0)
  verbose ? println("$Δt seconds, iter = $it") : nothing
  iter["amg_0-pcg"][1] = it

  if do_lorasc_0_pcg
    print("lorasc$(ndom)_0-pcg solve of A * u = b ...")
    Δt = @elapsed _, it, _ = pcg(A, b, M=Π_lorasc_0)
    verbose ? println("$Δt seconds, iter = $it") : nothing
  end

  print("lorasc_0-eigpcg solve of A * u = b ...")
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
  iter["bj$(nbj)_0-eigdefpcg"][1] = it

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

    verbose ? print("amg_0-pcg of A * u = b ... ") : nothing
    Δt = @elapsed _, it, _ = pcg(A, b, M=Π_amg_0)
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["amg_0-pcg"][s] = it

    if do_lorasc_0_pcg
      print("lorasc$(ndom)_0-pcg solve of A * u = b ...")
      Δt = @elapsed _, it, _ = pcg(A, b, M=Π_lorasc_0)
      verbose ? println("$Δt seconds, iter = $it") : nothing
    end

    print("lorasc$(ndom)_0-eigdefpcg solve of A * u = b ...")
    x .= 0.
    Δt = @elapsed _, it, _, W_lorasc = eigdefpcg(A, b, x, Π_lorasc_0, W_lorasc, spdim)
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["lorasc$(ndom)_0-eigdefpcg"][s] = it

    print("bj$(nbj)_0-eigdefpcg solve of A * u = b ...")
    x .= 0.
    Δt = @elapsed _, it, _, W_bj = eigdefpcg(A, b, x, Π_bj_0, W_bj, spdim)
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["bj$(nbj)_0-eigdefpcg"][s] = it

    print("chol16_0-eigdefpcg solve of A * u = b ...")
    x .= 0.
    Δt = @elapsed _, it, _, W_chol16 = eigdefpcg(A, b, x, Π_chol16_0, W_chol16, spdim)
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter["chol16_0-eigdefpcg"][s] = it
  end

  npzwrite("data/test01_$(root_fname)_amg_0-pcg.it.npz",
           iter["amg_0-pcg"])
  npzwrite("data/test01_$(root_fname)_lorasc$(ndom)_0-eigdefpcg_nvec$(nvec)_sdpim$(spdim).it.npz",
           iter["lorasc$(ndom)_0-eigdefpcg"])
  npzwrite("data/test01_$(root_fname)_bj$(nbj)_0-eigdefpcg_nvec$(nvec)_sdpim$(spdim).it.npz",
           iter["bj$(nbj)_0-eigdefpcg"])
  npzwrite("data/test01_$(root_fname)_chol16_0-eigdefpcg_nvec$(nvec)_sdpim$(spdim).it.npz",
           iter["chol16_0-eigdefpcg"])
end

test01(Π_amg_0,
       Π_lorasc_0,
       Π_bj_0,
       Π_chol16_0,
       sampler,
       verbose,
       do_lorasc_0_pcg,
       nsmp)