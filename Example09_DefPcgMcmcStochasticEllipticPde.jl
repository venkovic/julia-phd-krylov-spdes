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
import JLD

using LinearAlgebra: isposdef, rank

troubleshoot = true

maxit = 5_000
tentative_nnode = 20_000
load_existing_mesh = false

ndom = 8
load_existing_partition = false

nbj = ndom

nvec = floor(Int, 1.25 * ndom)
spdim = 3 * ndom

const preconds_with_dd = ("lorasc$(ndom)_0", "lorasc$(ndom)_1", "neumann-neumann$(ndom)_0")

preconds = ["neumann-neumann$(ndom)_0"] # ∈ ("amg_0", "bj$(nbj)_0") ⋃ preconds_with_dd 

nchains = 50
nsmp = 5
seed!(481_456)

model = "SExp"
sig2 = 1.
L = .1

root_fname = get_root_filename(model, sig2, L, tentative_nnode)

function f(x::Float64, y::Float64)
  return -1.
end
  
function uexact(xx::Float64, yy::Float64)
  return .734
end

#
# Load mesh
#
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

space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

#
# Load mesh parition
#
do_dd = false
for precond in preconds
  if precond in preconds_with_dd
    global do_dd = true
    break
  end
end

if do_dd
  if load_existing_partition
    epart, npart = load_partition(tentative_nnode, ndom)
  else
    epart, npart = mesh_partition(cells, ndom)
    save_partition(epart, npart, tentative_nnode, ndom)
  end

  ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l, node_owner,
  elemd, node_Γ, node_Γ_cnt, node_Id, nnode_Id = set_subdomains(cells,
                                                                cell_neighbors,
                                                                epart, 
                                                                npart,
                                                                dirichlet_inds_g2l)
end

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

Π = []
for precond in preconds

  if precond == "amg_0"
    printlnln("prepare amg_0 preconditioner for A_0 ...")
    Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);
    push!(Π, Π_amg_0)
    flush(stdout)
  
  elseif precond == "lorasc$(ndom)_0"
    printlnln("prepare_lorasc_precond for A_0 with ε = 0 ...")
    Π_lorasc_0 = @time prepare_lorasc_precond(ndom,
                                             cells,
                                          points,
                                          epart,
                                          ind_Id_g2l,
                                          ind_Γd_g2l,
                                          ind_Γ_g2l,
                                          ind_Γd_Γ2l,
                                          node_owner,
                                          node_Γ_cnt,
                                          exp.(g_0),
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          f,
                                          uexact,
                                          do_local_schur_assembly=true,
                                          low_rank_correction=:exact,
                                          ε=.01)
    push!(Π, Π_lorasc_0)
    flush(stdout)

  elseif precond == "lorasc$(ndom)_1"
    printlnln("prepare_lorasc_precond for A_0 with ε = 0.01 ...")
    Π_lorasc_1 = @time prepare_lorasc_precond(ndom,
                                          cells,
                                          points,
                                          epart,
                                          ind_Id_g2l,
                                          ind_Γd_g2l,
                                          ind_Γ_g2l,
                                          ind_Γd_Γ2l,
                                          node_owner,
                                          node_Γ_cnt,
                                          exp.(g_0),
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          f,
                                          uexact,
                                          do_local_schur_assembly=true,
                                          low_rank_correction=:exact,
                                          ε=0)
    push!(Π, Π_lorasc_1)
    flush(stdout)

  elseif precond == "bj$(nbj)_0"
    printlnln("prepare bj_0 preconditioner for A_0 ...")
    Π_bj_0 = @time BJPreconditioner(nbj, A_0);
    push!(Π, Π_bj_0)
    flush(stdout)

  elseif precond == "chol16"
    printlnln("prepare chol16_0 preconditioner for A_0 ...")
    Π_chol16_0 = @time get_cholesky16(A_0);
    push!(Π, Π_chol16_0)
    flush(stdout)

  elseif precond == "neumann-neumann$(ndom)_0"
    printlnln("prepare neumann-neummann preconditioner for A_0 ...")
    ΠS_nn_0, Π_IId = prepare_neumann_neumann_schur_precond(ndom,
                                                              cells,
                                                              points,
                                                              epart,
                                                              ind_Id_g2l,
                                                              ind_Γd_g2l,
                                                              ind_Γ_g2l,
                                                              ind_Γd_Γ2l,
                                                              node_owner,
                                                              node_Γ_cnt,
                                                              exp.(g_0),
                                                              f,
                                                              uexact,
                                                              load_partition=false)
    push!(Π, ΠS_nn_0)
    flush(stdout)
  end
end

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
                                      spdim::Int,
                                      maxit::Int;
                                      do_pcg=true,
                                      do_eigpcg=true,
                                      do_eigdefpcg=true,
                                      do_defpcg=true,
                                      verbose=true,
                                      save_results=false,
                                      troubleshoot=false)

  status = 0 

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

  do_schur = false
  for precond in preconds
    occursin("neumann-neumann", precond) ? do_schur = true : nothing
  end

  if do_schur
    S, b_schur = do_condensed_isotropic_elliptic_assembly(ndom,
                                                          cells,
                                                          points,
                                                          epart,
                                                          ind_Id_g2l,
                                                          ind_Γd_g2l,
                                                          ind_Γ_g2l,
                                                          ind_Γd_Γ2l,
                                                          node_owner,
                                                          node_Γ_cnt,
                                                          exp.(sampler.g),
                                                          f,
                                                          uexact,
                                                          Π_IId)
    n_Γ, = size(b_schur)
    x_schur = zeros(Float64, n_Γ)
    WtS = Array{Float64,2}(undef, nvec, n_Γ)
  end


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
  it = 0
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

      if do_schur
        S, b_schur = do_condensed_isotropic_elliptic_assembly(ndom,
                                                              cells,
                                                              points,
                                                              epart,
                                                              ind_Id_g2l,
                                                              ind_Γd_g2l,
                                                              ind_Γ_g2l,
                                                              ind_Γd_Γ2l,
                                                              node_owner,
                                                              node_Γ_cnt,
                                                              exp.(sampler.g),
                                                              f,
                                                              uexact,
                                                              Π_IId)
      end
    end

    for (p, precond) in enumerate(preconds)

      if do_pcg
        method = precond * "-pcg"
        verbose ? print("$method of A * u = b ... ") : nothing
        troubleshoot ? save_system(A, b) : nothing 
        try
          if occursin("neumann-neumann", precond)
            x_schur .= 0
            Δt = @elapsed _, it, _ = pcg(S, b_schur, x_schur, Π[p], maxit=maxit)
          else
            x .= 0
            Δt = @elapsed _, it, _  = pcg(A, b, x, Π[p], maxit=maxit)
          end
        catch err
          status = -1
          return iter, status
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
        flush(stdout)
      end

      if do_eigpcg
        method = precond * "-eigpcg"
        verbose ? print("$method of A * u = b ... ") : nothing
        troubleshoot ? save_system(A, b) : nothing 
        try
          if occursin("neumann-neumann", precond)
            
            if s == 1
              x_schur .= 0
            else
              WtS .= W[method]'S
              H = WtS * W[method]
              x_schur .= W[method] * (H \ (W[method]'b_schur))
            end
            
            Δt = @elapsed _, it, _, W[method] = eigpcg(S, b_schur, x_schur, Π[p], nvec, spdim, maxit=maxit)
          else
            
            if s == 1
              x .= 0
            else
              WtA .= W[method]'A
              H = WtA * W[method]
              x .= W[method] * (H \ (W[method]'b))
            end
            
            Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim, maxit=maxit)
          end
        catch err
          status = -1
          return iter, status
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
        flush(stdout)
      end

      if do_eigdefpcg
        method = precond * "-eigdefpcg"
        verbose ? print("$method of A * u = b ... ") : nothing
        if s == 1
          troubleshoot ? save_system(A, b) : nothing

          #Δt = @elapsed _, it, _, W[method] = eigpcg(S, b_schur, x, Π[p], nvec, spdim, maxit=maxit)
          try
            if occursin("neumann-neumann", precond)
              x_schur .= 0
              Δt = @elapsed _, it, _, W[method] = eigpcg(S, b_schur, x_schur, Π[p], nvec, spdim, maxit=maxit)
            else
              x .= 0
              Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim, maxit=maxit)
            end
          catch err
            status = -1
            println(err)
            return iter, status
          end
        else
          troubleshoot ? save_deflated_system(A, b, W[method]) : nothing
          try
            if rank(W[method]) < .9 * nvec
              status = -1
              return iter, status
            end
            if occursin("neumann-neumann", precond)
              x_schur .= 0
              Δt = @elapsed _, it, _, W[method] = eigdefpcg(S, b_schur, x_schur, Π[p], W[method], spdim, maxit=maxit)
            else
              x .= 0
              Δt = @elapsed _, it, _, W[method] = eigdefpcg(A, b, x, Π[p], W[method], spdim, maxit=maxit)
            end
          catch err
            status = -1
            return iter, status
          end
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
        flush(stdout)
      end

      if do_defpcg
        method = precond * "-defpcg"
        x .= 0
        verbose ? print("$method of A * u = b ... ") : nothing
        if s == 1
          # Some work remains to do here
          troubleshoot ? save_system(A, b) : nothing 
          try
            if occursin("neumann-neumann", precond)
              x_schur .= 0
              Δt = @elapsed _, it, _, W[method] = eigpcg(S, b_schur, x_schur, Π[p], nvec, spdim, maxit=maxit)
            else
              x .= 0
              Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim, maxit=maxit)
            end
          catch err
            status = -1
            return iter, status
          end
        else
          troubleshoot ? save_deflated_system(A, b, W[method]) : nothing 
          try
            if occursin("neumann-neumann", precond)
              x_schur .= 0
              Δt = @elapsed _, it, _, W[method] = defpcg(S, b_schur, W[method], x_schur, Π[p], maxit=maxit)
            else
              x .= 0
              Δt = @elapsed _, it, _, W[method] = defpcg(A, b, W[method], x, Π[p], maxit=maxit)
            end            
          catch err
            status = -1
            return iter, status
          end
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
        flush(stdout)
      end

    end # for (p, precond)

  end # for s in 1:nsmp

  if save_results
    for method in methods
      npzwrite("data/$(root_fname)_$method.it.npz", iter[method])
    end
  end

  return iter, status
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
                                        spdim::Int,
                                        maxit::Int;
                                        do_pcg=true,
                                        do_eigpcg=true,
                                        do_eigdefpcg=true,
                                        do_defpcg=true,
                                        verbose=true,
                                        save_results=true,
                                        troubleshoot=false)

  iters = Dict{String,Array{Int,2}}()

  ichain = 1
  while ichain <= nchains

    println("\n\nworking on chain $ichain / $nchains ...")

    iter, status = test_solvers_on_single_chain(nsmp, Λ, Ψ, Π, preconds, nvec, spdim, maxit,
                                                do_pcg=do_pcg,
                                                do_eigpcg=do_eigpcg,
                                                do_eigdefpcg=do_eigdefpcg,
                                                do_defpcg=do_defpcg,
                                                verbose=verbose,
                                                troubleshoot=troubleshoot)
    
    if status == 0
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
      ichain += 1
    end # if status = 0
  end

  return iters
end

iters = test_solvers_on_several_chains(nchains, nsmp, Λ, Ψ, Π,
                                       preconds, nvec, spdim, maxit,
                                       do_pcg=true,
                                       do_eigpcg=false,
                                       do_eigdefpcg=false,
                                       do_defpcg=false,
                                       save_results=false,
                                       troubleshoot=troubleshoot)


# get_constant_preconds()
# get_constant_preconds_with_dd()
# test_solvers_on_single_chain()
# test_solvers_on_single_chain()
# test_solvers_on_several_chains()
