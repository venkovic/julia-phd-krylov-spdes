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

#using Distributed
#using SharedArrays
#addprocs(11)


using Clustering
using Distributions: MvNormal, Normal, cdf
using Statistics: quantile
using LazySets: convex_hull
using LinearAlgebra: isposdef, rank

include("Example18_Clustering2D_Functions.jl")

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
ns = 100_000
Ps = (10, 100, 1_000, 10_000)
distances = ("L2", "cdf",)
nKL = 2
Λs = [[1., 1.],
      [1., .1],
      [1., .01]]


hηt_kmeans, hξt_kmeans, freqs_kmeans, w2_kmeans = do_kmeans(ns, Ps, distances, nKL, Λs)


for P in Ps
  for dist in distances
    for Λ in Λs
      npzwrite("data/Example18_$(P)_$(dist)_$(Λ[2]).freqs_kmeans.npz", freqs_kmeans[P][dist][Λ])
      npzwrite("data/Example18_$(P)_$(dist)_$(Λ[2]).centers.npz", hξt_kmeans[P][dist][Λ])
      npzwrite("data/Example18_$(P)_$(dist)_$(Λ[2]).w2_kmeans.npz", w2_kmeans[P][dist][Λ])
    end
  end
end

Ps = (10, 100, 1_000)

n = 1_000
x = LinRange(-4.05, 4.05, n)

#L2_map = SharedArray{Float64}(n, n)
#cluster_map = SharedArray{Int}(n, n)
#freq_map = SharedArray{Float64}(n, n)

L2_map = Array{Float64,2}(undef, n, n)
cluster_map = Array{Int,2}(undef, n, n)
freq_map = Array{Float64,2}(undef, n, n)

for P in Ps
  for dist in distances
    for Λ in Λs
      #@distributed for j in 1:n
      println("computing distortion map for Λ[2] = $(Λ[2]), dist = $dist, P = $P.")
      pixels = [Array{Array{Float64,1},1}([]) for p in 1:P]
      for j in 1:n
        for i in 1:n
          ξ = [x[i], x[j]]
          if dist == "L2"
            η = ξ .* sqrt.(Λ)
          elseif dist == "cdf"
            η = cdf.(Normal(), ξ)
            η .*= sqrt.(Λ)
          end
          #hη = hηt_kmeans[P][dist][Λ][:, 1]
          pmin = 1
          dmin = (η - hηt_kmeans[P][dist][Λ][:, 1])'*(η - hηt_kmeans[P][dist][Λ][:, 1]) 
          for p in 2:P
            d = (η - hηt_kmeans[P][dist][Λ][:, p])'*(η - hηt_kmeans[P][dist][Λ][:, p])
            if d < dmin
              #hη .= hηt_kmeans[P][dist][Λ][:, p]
              pmin = p
              dmin = d
            end
          end
          L2_map[i, j] = dmin
          cluster_map[i, j] = pmin
          freq_map[i, j] = freqs_kmeans[P][dist][Λ][pmin]
          if length(pixels[pmin]) == 0
            pixels[pmin] = [[x[i]; x[j]]]
          else
            push!(pixels[pmin], [x[i]; x[j]])
          end
        end
      end

      bnd_hull = []
      for p in 1:P
        hull = convex_hull(pixels[p])
        hull_cast = Array{Float64,2}(undef, 2, length(hull)[])
        for (i, vec) in enumerate(hull)
          hull_cast[:, i] = vec
        end
        push!(bnd_hull, hull_cast)
      end

      npzwrite("data/Example18_$(P)_$(dist)_$(Λ[2]).L2_map.npz", L2_map)
      npzwrite("data/Example18_$(P)_$(dist)_$(Λ[2]).freq_map.npz", freq_map)
      npzwrite("data/Example18_$(P)_$(dist)_$(Λ[2]).cluster_map.npz", cluster_map)
      for p in 1:P
        npzwrite("data/Example18_$(P)_$(dist)_$(Λ[2]).$p.bnd_hull.npz", bnd_hull[p])
      end
    end
  end
end