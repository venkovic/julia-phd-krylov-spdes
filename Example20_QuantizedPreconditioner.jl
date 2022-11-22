push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./Utils/")
push!(LOAD_PATH, "./MyPreconditioners/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: cg, pcg, defpcg

using Utils: space_println, printlnln

import SuiteSparse
import LinearAlgebra
import AlgebraicMultigrid
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using NPZ: npzread, npzwrite
using Random: seed!; seed!(123_456);
using LinearMaps: LinearMap, cholesky
using SparseArrays: SparseMatrixCSC
using Distributions: Normal

include("Example20_QuantizedPreconditioner_Functions.jl")

tentative_nnode = 100_000
load_existing_mesh = false

nreals = 100_000

model = "SExp"
sig2 = 1.
L = .1
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

precond = "k-means" # precond ∈ ("k-means", "deterministic-grid")

if precond == "k-means"
  ns = 100_000
  P = 10 # P ∈ (10, 100, 1_000, 10_000,)
  dist = "L2" # dist ∈ ("L2", "cdf",)
  nKL_trunc = 8 # _nKL ∈ [8, 24, 48, 170]
elseif precond == "deterministic-grid"
  m = 1 # m ∈ (0, 1, ...,)
  P = 1 + 2^m
  nKL_trunc = m
end

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

function f(x::Float64, y::Float64)
  return -1.
end
  
function uexact(xx::Float64, yy::Float64)
  return .734
end

println()
space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
nKL = length(Λ)

# Define quantization
if precond == "k-means"
  hηt_kmeans, hξt_kmeans = do_kmeans(ns, P, dist, nKL_trunc, Λ)
elseif precond == "deterministic-grid"
  hηt_kmeans, hξt_kmeans = get_deterministic_grid(P, nKL_trunc, Λ)
end

# Sample and attribute realizations
Ξ = rand(Normal(), nKL, nreals)
which = Array{Int,1}(undef, nreals)
for ireal in 1:nreals  
  which[ireal] = find_precond(scale_data(Ξ[:, ireal], Λ, dist), hηt_kmeans)
end

# Save realizations
for p in 1:P
  if precond == "k-means"
    npzwrite("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.reals.npz", Ξ[:, which .== p])
  elseif precond == "deterministic-grid"
    npzwrite("data/Example20.m$m.P$P.p$p.reals.npz", Ξ[:, which .== p])
  end
end

# Test preconditioners
ξ = Array{Float64,1}(undef, nKL)
for p in 1:P
  ξ[1:nKL_trunc] = hξt_kmeans[:, p]
  ξ[nKL_trunc+1:end] .= 0
  set!(Λ, Ψ, ξ, g)

  printlnln("do_isotropic_elliptic_assembly for preconditioner p = $p / $P ...")
  hAt, _ = @time do_isotropic_elliptic_assembly(cells, points,
                                                dirichlet_inds_g2l,                                                
                                                not_dirichlet_inds_g2l,
                                                point_markers,
                                                exp.(g), f, uexact)

  M_amg = AMGPreconditioner{SmoothedAggregation}(hAt)
  M_chol = cholesky(hAt)

  if precond == "k-means"
    Ξ = npzread("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.reals.npz")
  elseif
    Ξ = npzread("data/Example20.m$m.P$P.p$p.reals.npz")
  end
  _, nreals = size(Ξ)
  
  iters_amg = Array{Int,1}(undef, nreals)
  iters_chol = Array{Int,1}(undef, nreals)

  for ireal in 1:nreals
    set!(Λ, Ψ, Ξ[:, ireal], g)

    printlnln("do_isotropic_elliptic_assembly  ireal = $ireal / $nreals (p = $p / $P) ...")
    A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                                dirichlet_inds_g2l,
                                                not_dirichlet_inds_g2l,
                                                point_markers,
                                                exp.(g), f, uexact)
                                                
    printlnln("pcg solve of A * u = b with Π_amg ...")
    _, iters_amg[ireal], _ = @time pcg(A, b, zeros(A.n), M_amg)
    println("it = $(iters_amg[ireal])")
    
    printlnln("pcg solve of A * u = b with chol ...")
    _, iters_chol[ireal], _ = @time pcg(A, b, zeros(A.n), M_chol)
    println("it = $(iters_chol[ireal])")
  end
  
  if precond == "k-means"
    npzwrite("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.amg.iters.npz", iters_amg)
    npzwrite("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.chol.iters.npz", iters_chol)
  elseif precond == "deterministic-grid"
    npzwrite("data/Example20.m$m.P$P.p$p.amg.iters.npz", iters_amg)
    npzwrite("data/Example20.m$m.P$P.p$p.chol.iters.npz", iters_chol)
  end
end





