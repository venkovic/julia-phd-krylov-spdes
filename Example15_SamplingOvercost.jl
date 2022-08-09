import Pkg
using Distributed
Pkg.activate(".")

addprocs(10)

@everywhere begin
  push!(LOAD_PATH, "./Fem/")
  push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
  push!(LOAD_PATH, "./MyPreconditioners/")
  push!(LOAD_PATH, "./Utils/")
  import Pkg
  Pkg.activate(".")
end

@everywhere begin
  using Fem
  using RecyclingKrylovSolvers: cg, pcg, defpcg, eigpcg, eigdefpcg
  using Preconditioners: AMGPreconditioner, SmoothedAggregation
  using MyPreconditioners: BJPreconditioner

  using Utils: space_println, printlnln, 
               save_deflated_system, save_system,
               load_system 

  using NPZ: npzread, npzwrite
  using Random: seed!
  using LinearMaps: LinearMap
  import SuiteSparse
  import JLD

  using LinearAlgebra: isposdef, rank

  include("Example15_SamplingOvercost_Functions.jl")
end

@everywhere begin 
  maxit = 5_000
  tentative_nnode = 16_000 # 4_000, 8_000, 16_000, 32_000, 64_000, 128_000
  load_existing_mesh = false
end

preconds = ["amg_0"] # ∈ ("amg_0", 
                     #    "bj$(nbj)_0",
                     #    "lorasc$(ndom)_1")

@everywhere begin
  nsmp = 10_000
  n_mcmc = 500 # ∈ {50, 100, 150, 200, 250, 300, 350, 400, 450, 500}
end
seed!(481_456)

@everywhere begin
  model = "SExp"
  sig2 = 1.
  L = .1

  root_fname = get_root_filename(model, sig2, L, tentative_nnode)

  function f(x::Float64, y::Float64)
    return -1.
  end
  
  function uexact(x::Float64, y::Float64)
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
end

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

#
# Load kl representation
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
@everywhere begin
  Λ = $Λ
  Ψ = $Ψ
end

#
# Prepare constant preconditioners
#
Π0 = get_constant_preconds(A_0)
@everywhere Π0 = $Π0

#
# Get DoF index of closest vertex in the mesh
#
index = find_index_of_closest_vertex(.5, .5, points, not_dirichlet_inds_g2l)
@everywhere index = $index

#
# Run test
#
@spawnat 1 begin
  u_values_mc = solve_mc_sample(nsmp, Λ, Ψ, Π0, maxit, index)
  npzwrite("data/$root_fname.u_values_mc.npz", u_values_mc)
end

for worker in workers()
  @spawnat worker begin
    u_values_mcmc, repeats = solve_single_chain(nsmp, Λ, Ψ, n_mcmc, Π0, maxit, index)
    if n_mcmc >= length(Λ)
      npzwrite("data/$root_fname.u_values_mcmc.chain$(myid()-2).npz", u_values_mcmc)
      npzwrite("data/$root_fname.repeats_mcmc.chain$(myid()-2).npz", repeats)
    else
      npzwrite("data/$root_fname.u_values_hybrid$(n_mcmc).chain$(myid()-2).npz", u_values_mcmc)
      npzwrite("data/$root_fname.repeats_hybrid$(n_mcmc).chain$(myid()-2).npz", repeats)
    end
  end
end