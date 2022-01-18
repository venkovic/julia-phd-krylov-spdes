push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using Utils: space_println, printlnln

using NPZ: npzwrite

tentative_nnode = 4_000 # 40_000
load_existing_mesh = false

if load_existing_mesh
  cells, points, _, _ = load_mesh(tentative_nnode)
else
  mesh = get_mesh(tentative_nnode)
  save_mesh(mesh, tentative_nnode)
  cells = mesh.cell
  points = mesh.point
end

model = "SExp"
L = .1
sig2 = 1.
nev = 500

root_fname = get_root_filename(model, sig2, L, tentative_nnode)

function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  L = .1
  sig2 = 1.
  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

println("solve_kl ...")
Λ, Φ = @time solve_kl(cells, points, cov, nev, verbose=true)
npzwrite("data/$root_fname.kl-eigvals.npz", Λ)
npzwrite("data/$root_fname.kl-eigvecs.npz", Φ)

printlnln("sample ...")
ξ, g = @time draw(Λ, Φ)
npzwrite("data/$root_fname.greal.npz", g)