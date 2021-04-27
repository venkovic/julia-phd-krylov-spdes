push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./MyPreconditioners/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: cg, pcg
using Preconditioners: AMGPreconditioner, SmoothedAggregation
#using MyPreconditioners: BJPreconditioner, BJop,
#                         Cholesky16, get_cholesky16,
#                         Cholesky32, get_cholesky32

using Utils: space_println, printlnln, 
             save_deflated_system, save_system,
             load_system 

using NPZ: npzread, npzwrite
using Random: seed!
using LinearMaps: LinearMap
import SuiteSparse
import JLD

using LinearAlgebra: isposdef, rank

include("Example12_Quantization_Functions.jl")

maxit = 5_000
tentative_nnode = 4_000
load_existing_mesh = false

nsmp = 10_000
seed!(481_456)

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

space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

"""
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
"""

#
# Load kl representation
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")


#
# Get Voronoi quantizer
#
nsmp_preconds = 10_000
P = 5 # 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
distance = "cdf-full"
X, centroids, assignments, costs = get_quantizer(nsmp_preconds, P, Λ, distance=distance)

#
# Prepare constant preconditioners
#
Π = get_centroidal_preconds(centroids, Λ, Ψ)


#
# Test solver with quantized preconditioner
#
nsmp = 1_000
assignments, iters, dists, dists_to_0 = test_solver_with_centroidal_preconds(nsmp, Π, centroids, Λ, Ψ)

#
# Save data
#
npzwrite("data/Example12_$(root_fname)_$(distance)_$(P)_$(nsmp_preconds).assignments.npz", assignments)
npzwrite("data/Example12_$(root_fname)_$(distance)_$(P)_$(nsmp_preconds).iters.npz", iters)
npzwrite("data/Example12_$(root_fname)_$(distance)_$(P)_$(nsmp_preconds).dists.npz", dists)
npzwrite("data/Example12_$(root_fname)_$(distance)_$(P)_$(nsmp_preconds).dists_to_0.npz", dists_to_0)
