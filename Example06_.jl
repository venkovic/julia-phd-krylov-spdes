push!(LOAD_PATH, "./Fem/")
import Pkg
Pkg.activate(".")
using Fem
using NPZ


model = "SExp"
sig2 = 1.
L = .1
tentative_nnode = 20_000

root_fname = get_root_filename(model, sig2, L, tentative_nnode)

cells = npzread("data/DoF$tentative_nnode.cells.npz") .+ 1
points = npzread("data/DoF$tentative_nnode.points.npz")

M = get_mass_matrix(cells', points')
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
g = npzread("data/$root_fname.kl-eigvecs.npz")
ξ, g = draw(Λ, Ψ)

χ = get_kl_coordinates(g, Λ, Ψ, M)


using Statistics
nsim = 10
mu = 0.
for i in 1:nsim
  draw!(Λ, Ψ, ξ, g)
  global mu = ((i - 1) * mu + mean(g)) / i
  println("$i, $mu")
end

# Compare χ and ξ
# Assemble (FAST)
# Apply BCs (FAST)
# Cg solve
# Preconditioning