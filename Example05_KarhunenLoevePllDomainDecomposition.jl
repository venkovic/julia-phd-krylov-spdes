using Distributed

#addprocs(([("marcel", 4)]), tunnel=true)
#addprocs(([("hector", 6)]), tunnel=true)
#addprocs(([("lucien", 8)]), tunnel=true, max_parallel=30)
addprocs(([("lucien", 3)]), tunnel=true, max_parallel=30)
#addprocs(([("andrew", 3)]), tunnel=true)
#addprocs(([("venkovic@moorcock", 6)]), tunnel=true,
#             dir="/home/venkovic/Dropbox/Git/julia-fem/",
#             exename="/home/venkovic/julia-1.5.3/bin/julia")
addprocs(3) # Add procs after remote hosts due to issue with ClusterManagers

@everywhere begin
  push!(LOAD_PATH, "./Fem/")
  push!(LOAD_PATH, "./Utils/")
end

@everywhere begin
  import Pkg
  Pkg.activate(".")
end

using Utils: space_println, printlnln

@everywhere begin 
  using Fem
  using Distributed
end

@everywhere worker_timeout() = 10_000.

using NPZ: npzwrite

@everywhere begin
  ndom = 20
  nev = 40
  tentative_nnode = 20_000
  forget = 1e-6
end

load_existing_mesh = false
load_existing_partition = false

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

# Broadcast mesh data
@everywhere begin
  nnode = $nnode
  cells = $cells
  points = $points
  epart = $epart
end

@everywhere function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  L = .1
  sig2 = 1.
  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

model = "SExp"
sig2 = 1.
L = .1
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

printlnln("pll_solve_local_kl ...")
@time domain = @sync @distributed merge! for idom in 1:ndom
  relative_local, _ = suggest_parameters(nnode)
  pll_solve_local_kl(cells, points, epart, cov, nev, idom, 
                     relative=relative_local)
end
println("... done with pll_solve_local_kl.")

energy_expected = 0.
for idom in 1:ndom
  global energy_expected += domain[idom].energy
end

# Count number of local modes retained in each subdomain
md = zeros(Int, ndom) 
for idom in 1:ndom
  md[idom] = size(domain[idom].ϕ)[2]
end

# Broadcast numbers of local modes retained
@everywhere md = $md

printlnln("pll_do_global_mass_covariance_reduced_assembly ...")
@time begin
  K = @sync @distributed (+) for idom in 1:ndom
  pll_do_global_mass_covariance_reduced_assembly(cells, points, 
                                                 domain, idom, md, cov,
                                                 forget=forget)
  end
end
println("... done with pll_do_global_mass_covariance_reduced_assembly.")

printlnln("solve_global_reduced_kl ...")
_, relative_global = suggest_parameters(nnode)
Λ, Ψ = @time solve_global_reduced_kl(nnode, K, energy_expected, domain, 
                                     relative=relative_global)
println("... done with do_global_mass_covariance_reduced_assembly.")
npzwrite("data/$root_fname.kl-eigvals.npz", Λ)
npzwrite("data/$root_fname.kl-eigvecs.npz", Ψ)

printlnln("sample ...")
ξ, g = @time draw(Λ, Ψ)

printlnln("sample in place ...")
@time draw!(Λ, Ψ, ξ, g)
npzwrite("data/$root_fname.greal.npz", g)