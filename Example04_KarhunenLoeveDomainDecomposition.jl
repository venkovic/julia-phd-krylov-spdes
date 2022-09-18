push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using Utils: space_println, printlnln

using NPZ: npzwrite

tentative_nnode = 8_000 # 4_000, 100_000, 8_000, 16_000, 32_000, 64_000, 128_000
load_existing_mesh = false

ndom = 20 # 5, 10, 20, 30, 80, 200
load_existing_partition = false

nev = 70 # 25
forget = 1e-6

if load_existing_mesh
  cells, points, point_markers, cell_neighbors = load_mesh(tentative_nnode)
  _, nnode = size(points)
else
  mesh = get_mesh(tentative_nnode)
  save_mesh(mesh, tentative_nnode)
  cells = mesh.cell
  points = mesh.point
  point_markers = mesh.point_marker
  cell_neighbors = mesh.cell_neighbor
  _, nnode = size(points)
end

dirichlet_inds_g2l, not_dirichlet_inds_g2l,
dirichlet_inds_l2g, not_dirichlet_inds_l2g = 
get_dirichlet_inds(points, point_markers)

if load_existing_partition
  epart, npart = load_partition(tentative_nnode, ndom)
else
  epart, npart = mesh_partition(cells, ndom)
  save_partition(epart, npart, tentative_nnode, ndom)
end

model = "SExp"
L = .1 # .05, .1, .5, 1., 5., 10.
sig2 = 1.
save_eigvals = true
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  L = .1
  sig2 = 1.
  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

relative_local, relative_global = suggest_parameters(nnode)

inds_g2ld = [Dict{Int,Int}() for _ in 1:ndom]
inds_l2gd = Array{Int,1}[]
elemsd = Array{Int,1}[]
Φd = Array{Float64,2}[]
centerd = Array{Float64,1}[]
energy_expected = 0.

space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

printlnln("solve_local_kl ...")
@time for idom in 1:ndom
  subdom = solve_local_kl(cells, points, epart, cov, nev, idom, relative=relative_local,
                           save_eigvals=save_eigvals, model=model, sig2=sig2, L=L, 
                           tentative_nnode=tentative_nnode)
  inds_g2ld[idom] = subdom.inds_g2l
  push!(inds_l2gd, subdom.inds_l2g)
  push!(elemsd, subdom.elems)
  push!(Φd, subdom.Φ)
  push!(centerd, subdom.center)
  global energy_expected += subdom.energy
end
println("... done with solve_local_kl.")

printlnln("do_global_mass_covariance_reduced_assembly ...")
K = @time do_global_mass_covariance_reduced_assembly(cells, points, elemsd,
                                                     inds_g2ld, inds_l2gd, Φd,
                                                     centerd, cov, forget=forget)
println("... done with do_global_mass_covariance_reduced_assembly.")

printlnln("solve_global_reduced_kl ...")
Λ, Ψ = @time solve_global_reduced_kl(nnode, K, energy_expected,
                                     elemsd, inds_l2gd, Φd,
                                     relative=relative_global)
npzwrite("data/$root_fname.kl-eigvals.npz", Λ)
npzwrite("data/$root_fname.kl-eigvecs.npz", Ψ)

printlnln("sample ...")
ξ, g = @time draw(Λ, Ψ)

printlnln("sample in place ...")
@time draw!(Λ, Ψ, ξ, g)
npzwrite("data/$root_fname.greal.npz", g)


