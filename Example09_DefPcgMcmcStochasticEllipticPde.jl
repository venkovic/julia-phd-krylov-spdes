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

using Utils: space_println, printlnln, 
             save_deflated_system, save_system,
             load_system 

using NPZ: npzread, npzwrite
using Random: seed!
using LinearMaps: LinearMap
import SuiteSparse

tentative_nnode = 20_000
load_existing_mesh = false

ndom = 8
load_existing_partition = false

nbj = ndom

nvec = floor(Int, 1.25 * ndom)
spdim = 3 * ndom

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
flush(stdout)

printlnln("prepare_lorasc_precond for A_0 with ε = 0 ...")
Π_lorasc_0 = @time prepare_lorasc_precond(tentative_nnode,
                                          ndom,
                                          cells,
                                          points,
                                          cell_neighbors,
                                          exp.(g_0),
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          f,
                                          uexact,
                                          do_local_schur_assembly=true,
                                          low_rank_correction=:exact,
                                          ε=0)
flush(stdout)

printlnln("prepare_lorasc_precond for A_0 with ε = 0.01 ...")
Π_lorasc_1 = @time prepare_lorasc_precond(tentative_nnode,
                                          ndom,
                                          cells,
                                          points,
                                          cell_neighbors,
                                          exp.(g_0),
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          f,
                                          uexact,
                                          do_local_schur_assembly=true,
                                          low_rank_correction=:exact,
                                          ε=.01)
flush(stdout)

printlnln("prepare bj_0 preconditioner for A_0 ...")
Π_bj_0 = @time BJPreconditioner(nbj, A_0);
flush(stdout)

printlnln("prepare chol16_0 preconditioner for A_0 ...")
Π_chol16_0 = @time get_cholesky16(A_0);
flush(stdout)

#
# Load kl representation and prepare mcmc sampler
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")


"""
     test_solvers_on_a_chain(nsmp::Int,
                             Λ::Array{Float64,1},
                             Ψ::Array{Float64,2}
                             Π,
                             precond_labels::Array{String,1},
                             do_pcg::Bool,
                             do_eigpcg::Bool,
                             do_eigdefpcg::Bool,
                             do_defpcg::Bool,
                             nvec::Int,
                             spdim::Int;
                             verbose=true,
                             save_results=false)


"""
function test_solvers_on_single_chain(nsmp::Int,
                                      Λ::Array{Float64,1},
                                      Ψ::Array{Float64,2},
                                      Π,
                                      preconds::Array{String,1},
                                      nvec::Int,
                                      spdim::Int;
                                      do_pcg=true,
                                      do_eigpcg=true,
                                      do_eigdefpcg=true,
                                      do_defpcg=true,
                                      verbose=true,
                                      save_results=false)

  sampler = prepare_mcmc_sampler(Λ, Ψ)

  verbose ? println("\n1 / $nsmp") : nothing
  verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
  Δt = @elapsed A, b = do_isotropic_elliptic_assembly(cells, points,
                                                      dirichlet_inds_g2l,
                                                      not_dirichlet_inds_g2l,
                                                      point_markers,
                                                      exp.(sampler.g),
                                                      f, uexact)
  verbose ? println("$Δt seconds") : nothing

  methods = String[]
  W = Dict{String,Array{Float64,2}}()
  WtA = Array{Float64,2}(undef, nvec, A.n)
  iter = Dict{String,Array{Int,1}}()

  for precond in preconds
    if do_pcg
      method = precond * "-pcg"
      push!(methods, method)
      iter[method] = Array{Int,1}(undef, nsmp)
    end
    
    if do_eigpcg 
      method = precond * "-eigpcg"
      push!(methods, method)
      iter[method] = Array{Int,1}(undef, nsmp)
      W[method] = Array{Float64,2}(undef, A.n, nvec)
    end

    if do_eigdefpcg
      method = precond * "-eigdefpcg"
      push!(methods, method)
      iter[method] = Array{Int,1}(undef, nsmp)
      W[method] = Array{Float64,2}(undef, A.n, nvec)
    end
    
    if do_defpcg
      method = precond * "-defpcg"
      push!(methods, method)
      iter[method] = Array{Int,1}(undef, nsmp)
      W[method] = Array{Float64,2}(undef, A.n, nvec)
    end
  end


  x = zeros(Float64, A.n)

  #
  # Sample ξ by mcmc and solve linear systems by def-pcg with 
  # online eigenvectors approximation
  #
  cnt_reals = 1
  for s in 1:nsmp

    if s > 1
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
    end

    for (p, precond) in enumerate(preconds)

      if do_pcg
        method = precond * "-pcg"
        x .= 0
        verbose ? print("$method of A * u = b ... ") : nothing
        Δt = @elapsed _, it, _  = pcg(A, b, x, Π[p])
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
      end

      if do_eigpcg
        method = precond * "-eigpcg"
        if s == 1
          x .= 0
        else
          WtA .= W[method]'A
          H = WtA * W[method]
          x .= W[method] * (H \ (W[method]'b))
        end
        verbose ? print("$method of A * u = b ... ") : nothing
        Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim)
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
      end

      if do_eigdefpcg
        method = precond * "-eigdefpcg"
        x .= 0
        verbose ? print("$method of A * u = b ... ") : nothing
        if s == 1
          Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim)
        else
          Δt = @elapsed _, it, _, W[method] = eigdefpcg(A, b, x, Π[p], W[method], spdim)
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
      end

      if do_defpcg
        method = precond * "-defpcg"
        x .= 0
        verbose ? print("$method of A * u = b ... ") : nothing
        if s == 1
          # Some work remains to do here
          Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim)
        else
          Δt = @elapsed _, it, _, W[method] = defpcg(A, b, W[method], x, Π[p])
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
      end

    end # for (p, precond)

  end # for s in 1:nsmp

  if save_results
    for method in methods
      npzwrite("data/$(root_fname)_$method.it.npz", iter[method])
    end
  end

  return iter
end


"""
     test_solvers_on_several_chains(nchains::Int,
                                    nsmp::Int,
                                    Λ::Array{Float64,1},
                                    Ψ::Array{Float64,2},
                                    Π,
                                    preconds::Array{String,1},
                                    do_pcg::Bool,
                                    do_eigpcg::Bool,
                                    do_eigdefpcg::Bool,
                                    do_defpcg::Bool,
                                    nvec::Int,
                                    spdim::Int;
                                    verbose=true,
                                    save_results=true)

"""
function test_solvers_on_several_chains(nchains::Int,
                                        nsmp::Int,
                                        Λ::Array{Float64,1},
                                        Ψ::Array{Float64,2},
                                        Π,
                                        preconds::Array{String,1},
                                        nvec::Int,
                                        spdim::Int;
                                        do_pcg=true,
                                        do_eigpcg=true,
                                        do_eigdefpcg=true,
                                        do_defpcg=true,
                                        verbose=true,
                                        save_results=true)

  iters = Dict{String,Array{Int,2}}()

  for ichain in 1:nchains

    println("\n\nworking on chain $ichain / $nchains ...")

    iter = test_solvers_on_single_chain(nsmp, Λ, Ψ, Π, preconds, nvec, spdim,
                                        do_pcg=do_pcg,
                                        do_eigpcg=do_eigpcg,
                                        do_eigdefpcg=do_eigdefpcg,
                                        do_defpcg=do_defpcg,
                                        verbose=verbose)

    for (method, _iter) in iter
      if haskey(iters, method) 
        iters[method] = hcat(iters[method], _iter)
      else
        iters[method] = reshape(_iter, length(_iter), 1)
      end
    end

    if save_results
      for (method, _iters) in iters
        if occursin("-eigpcg", method) | occursin("-eigdefpcg", method) | occursin("-defpcg", method)
          npzwrite("data/$(root_fname)_$(method)_nvec$(nvec)_spdim$(spdim).it.npz", _iters)
        else
          npzwrite("data/$(root_fname)_$(method).it.npz", _iters)
        end

      end
    end

    println("\n\n ... done working on chain $ichain / $nchains.")
  end

  return iters
end


#
# Is assemble_local_schurs necessary with ε = 0 
#

Π = [Π_amg_0, Π_lorasc_0, Π_lorasc_1, Π_bj_0]
preconds = ["amg_0",
            "lorasc$(ndom)_0",
            "lorasc$(ndom)_1",
            "bj$(nbj)_0"]
 
iters = test_solvers_on_several_chains(nchains, nsmp, Λ, Ψ, Π,
                                       preconds, nvec, spdim,
                                       do_pcg=false,
                                       do_eigpcg=true,
                                       do_eigdefpcg=true,
                                       do_defpcg=false,
                                       save_results=true)