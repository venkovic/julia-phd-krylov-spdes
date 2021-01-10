using Distributed

addprocs(([("marcel", 4)]), tunnel=true)
addprocs(([("andrew", 3)]), tunnel=true)
#addprocs(([("moorcock", 4)]), tunnel=true)
addprocs(3)

@everywhere begin
  push!(LOAD_PATH, "./Fem/")
  import Pkg
  Pkg.activate(".")
end

@everywhere begin 
  using Fem
  using Distributed
  using Distributions
  using DistributedOperations
end

import LinearAlgebra
using NPZ

# Advice:
# - At fixed value of ndom * nev, 
#   use large ndom and small nev.

@everywhere begin
  ndom = 400
  nev = 25
  tentative_nnode = 400_000
  forget = 1e-6
end

mesh = get_mesh(tentative_nnode)
npzwrite("data/DoF$(tentative_nnode).cells.npz", mesh.cell' .- 1)
npzwrite("data/DoF$(tentative_nnode).points.npz", mesh.point')
epart, npart = mesh_partition(mesh, ndom)
bcast(mesh, procs())
bcast(epart, procs())

@everywhere function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  L = .1
  sig2 = 1.
  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

model = "SExp"
sig2 = 1.
L = .1
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

# Distributed computation of local KL expansion for each subdomain  
@time domain = @sync @distributed merge! for idom in 1:ndom
  relative_local, _ = suggest_parameters(mesh.n_point)
  pll_solve_local_kl(mesh, epart, cov, nev, idom, 
                     relative=relative_local)
end

energy_expected = 0.
for idom in 1:ndom
  global energy_expected += domain[idom].energy
end

# Count number of local modes retained in each subdomain
md = zeros(Int, ndom) 
for idom in 1:ndom
  md[idom] = size(domain[idom].ϕ)[2]
end
bcast(md, procs())

# Do distributed assembly of globally reduced system
@time begin
  K = @sync @distributed (+) for idom in 1:ndom
  pll_do_global_mass_covariance_reduced_assembly(mesh.cell, mesh.point, 
                                                 domain, idom, md, cov,
                                                 forget=forget)
  end
end

# Solve globally reduced eigenvalue problem, and project eigenfunctions
_, relative_global = suggest_parameters(mesh.n_point)
Λ, Ψ = @time solve_global_reduced_kl(mesh, K, energy_expected, domain, 
                                     relative=relative_global)
npzwrite("data/$root_fname.kl-eigvals.npz", Λ)
npzwrite("data/$root_fname.kl-eigvecs.npz", Ψ)

# Sample
ξ, g = @time draw(Λ, Ψ)
@time npzwrite("data/$root_fname.greal.npz", g)