push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./MyPreconditioners/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: cg, pcg
using Preconditioners: AMGPreconditioner, SmoothedAggregation

using Utils: space_println, printlnln, 
             save_deflated_system, save_system,
             load_system 

using NPZ: npzread, npzwrite
using Random: seed!
using LinearMaps: LinearMap
import SuiteSparse
import JLD

using distributed

addprocs(11)


using Clustering
using Distributions: MvNormal, Normal, cdf
using Statistics: quantile

using LinearAlgebra: isposdef, rank

include("Example13_CLVQ_Functions.jl")

maxit = 5_000
tentative_nnode = 20_000
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


#
# Load kl representation
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")


#
# Get Voronoi quantizer by CLVQ and k-means
#
ns = 10_000
Ps = (10, 100, 1_000,) # 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
distances = ("L2", "cdf",)
nKL = 2

do_kmeans(ns, Ps, distances, nKL)

n = 10_000
x = LinRange(-4.05, 4.05, n)

w2_map = SharedArray{Float64,2}(n, n)
cluster_map = SharedArray{Int,2}(n, n)
freq_map = SharedArray{Float64,2}(n, n)

dt = @elapsed @distributed for j in 1:n
  for i in 1:n
    w2_map[i, j] = nothing
    cluster_map[i, j] = nothing
    freq_map[i, j] = nothing
  end
end