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
using Distributions: Normal, quantile
using LinearMaps: LinearMap, cholesky
using SparseArrays: SparseMatrixCSC
using Distributions: Normal

include("Example20_QuantizedPreconditioner_Functions.jl")

tentative_nnode = 100_000
load_existing_mesh = true

nreals = 100_000

model = "SExp"
sig2 = 1.
L = .1
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

precond = "k-means" # precond ∈ ("k-means", "deterministic-grid")

s = 2 * quantile(Normal(0, 1), 2/3)

if precond == "k-means"
  ns = 100_000
  P = 10_000 # P ∈ (10, 100, 1_000, 10_000,)
  dist = "cdf" # dist ∈ ("L2", "cdf",)
  nKL_trunc = 170 # nKL_trunc ∈ [8, 24, 48, 170]
elseif precond == "deterministic-grid"
  m = 0 # m ∈ (0, 1, ..., 13)
  if m == 0
    P = 1
  else
    P = 1 + 2^m
  end
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

M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
nKL = length(Λ)

# Define quantization
if precond == "k-means"
  hηt_kmeans, hξt_kmeans = do_kmeans(ns, P, dist, nKL_trunc, Λ)
elseif precond == "deterministic-grid"
  hηt_kmeans, hξt_kmeans = get_deterministic_grid(P, nKL_trunc, Λ, s)
end

# Sample and attribute realizations
Ξ = rand(Normal(), nKL, nreals)
which = Array{Int,1}(undef, nreals)
for ireal in 1:nreals
  if precond == "k-means"
    which[ireal] = find_precond(scale_data(Ξ[:, ireal], Λ, dist), hηt_kmeans)
  elseif precond == "deterministic-grid"
    which[ireal] = find_precond(scale_data(Ξ[:, ireal], Λ, "L2"), hηt_kmeans)
  end
end

# Save realizations
for p in 1:P
  if precond == "k-means"
    npzwrite("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.reals.npz", Ξ[:, which .== p])
  elseif precond == "deterministic-grid"
    npzwrite("data/Example20.m$m.P$P.p$p.reals.npz", Ξ[:, which .== p])
  end
end

ξ, g = draw(Λ, Ψ)


for p in 1:P
  ξ[1:nKL_trunc] = hξt_kmeans[:, p]
  ξ[nKL_trunc+1:end] .= 0
  set!(Λ, Ψ, ξ, g)

  hAt, _ = @time do_isotropic_elliptic_assembly(cells, points,
                                                dirichlet_inds_g2l,                                                
                                                not_dirichlet_inds_g2l,
                                                point_markers,
                                                exp.(g), f, uexact)

  M_amg = AMGPreconditioner{SmoothedAggregation}(hAt)
  M_chol = cholesky(hAt)

  Ξ = if precond == "k-means"
    npzread("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.reals.npz") 
  elseif precond == "deterministic-grid"
    npzread("data/Example20.m$m.P$P.p$p.reals.npz")
  end
  _, nreals = size(Ξ)
  
  iters_amg = Array{Int,1}(undef, nreals)
  iters_chol = Array{Int,1}(undef, nreals)

  for ireal in 1:nreals
    printlnln("working on realization $ireal / $nreals for p = $p / $P ...")
    @time begin
      set!(Λ, Ψ, Ξ[:, ireal], g)

      A, b = do_isotropic_elliptic_assembly(cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            exp.(g), f, uexact)
                                                
      _, iters_amg[ireal], _ = pcg(A, b, zeros(A.n), M_amg)
    
      _, iters_chol[ireal], _ = pcg(A, b, zeros(A.n), M_chol)
    end
  end
  
  if precond == "k-means"
    npzwrite("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.amg.iters.npz", iters_amg)
    npzwrite("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.chol.iters.npz", iters_chol)
  elseif precond == "deterministic-grid"
    npzwrite("data/Example20.m$m.P$P.p$p.amg.iters.npz", iters_amg)
    npzwrite("data/Example20.m$m.P$P.p$p.chol.iters.npz", iters_chol)
  end
end






