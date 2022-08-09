import Pkg
using Distributed
Pkg.activate(".")


push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./MyPreconditioners/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")


using Fem
using Utils: space_println, printlnln, 
             save_deflated_system, save_system,
             load_system 

using NPZ: npzread, npzwrite
using Random: seed!
using LinearMaps: LinearMap
import SuiteSparse
import JLD

using LinearAlgebra: isposdef, rank

 
tentative_nnode = 4_000 # 4_000, 8_000, 16_000, 32_000, 64_000, 128_000
load_existing_mesh = false

nsmp = (0, 10, 100, 1000)
seed!(481_456)

model = "SExp"
sig2 = 1.
L = .1

root_fname = get_root_filename(model, sig2, L, tentative_nnode)


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


#
# Load kl representation
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")


function sample_chain(M, Λ, Ψ)
  sampler = prepare_mcmc_sampler(Λ, Ψ)
  cnt_reals = 0
  reals = Dict()
  reals[0] = exp.(sampler.g)

  converged = false
  while !converged
    cnt_reals += draw!(sampler)
    for m in nsmp
      if !haskey(reals, m) && cnt_reals >= m
        reals[m] = exp.(sampler.g)
      end
    end
    if haskey(reals, nsmp[end])
      converged = true
    end
  end
  return reals
end

reals = sample_chain(M, Λ, Ψ)

for m in nsmp
  npzwrite("data/$root_fname.real_mcmc_$m.npz", reals[m])
end






















