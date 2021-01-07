push!(LOAD_PATH, "./Fem/")
import Pkg
Pkg.activate(".")
using Fem
using NPZ
import LinearAlgebra
import Arpack
using Distributions

# Advice:
# - At fixed value of ndom * nev, 
#   use large ndom and small nev.

ndom = 80
nev = 35
tentative_nnode = 20_000
forget = 1e-6

mesh = get_mesh(tentative_nnode)
npzwrite("data/DoF$tentative_nnode.cells.npz", mesh.cell' .- 1)
npzwrite("data/DoF$tentative_nnode.points.npz", mesh.point')
epart, npart = mesh_partition(mesh, ndom)

model = "SExp"
sig2 = 1.
L = .1
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  L = .1
  sig2 = 1.
  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

inds_g2ld = [Dict{Int,Int}() for _ in 1:ndom]
inds_l2gd = Array{Int,1}[]
elemsd = Array{Int,1}[]
ϕd = Array{Float64,2}[]
centerd = Array{Float64,1}[]
energy_expected = 0.

# Loop over subdomains and compute local KL expansions
@time for idom in 1:ndom
  subdom = solve_local_kl(mesh, epart, cov, nev, idom, relative=.996)
  inds_g2ld[idom] = subdom.inds_g2l
  push!(inds_l2gd, subdom.inds_l2g)
  push!(elemsd, subdom.elems)
  push!(ϕd, subdom.ϕ)
  push!(centerd, subdom.center)
  global energy_expected += subdom.energy
end

# Assemble globally reduced eigenvalue problem
K = @time do_global_mass_covariance_reduced_assembly(mesh.cell, mesh.point, elemsd,
                                                     inds_g2ld, inds_l2gd, ϕd,
                                                     centerd, cov, forget=forget)

# Solve globally reduced eigenvalue problem, and project eigenfunctions 
Λ, Ψ = @time solve_global_reduced_kl(mesh, K, energy_expected,
                                     elemsd, inds_l2gd, ϕd,
                                     relative=.995)
npzwrite("data/$root_fname.kl-eigvals.npz", Λ)
npzwrite("data/$root_fname.kl-eigvecs.npz", Ψ)

# Sample
ξ, g = @time draw(Λ, Ψ)
@time draw!(Λ, Ψ, ξ, g)
npzwrite("data/$root_fname.greal.npz", g)