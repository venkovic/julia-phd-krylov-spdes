push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: cg, pcg, defpcg
using Utils: space_println, printlnln

using Preconditioners: AMGPreconditioner, SmoothedAggregation
using NPZ: npzread
using Random: seed!; seed!(123_456);
using LinearMaps: LinearMap

tentative_nnode = 100_000
load_existing_mesh = false

ndom = 40
load_existing_partition = false

model = "SExp"
sig2 = 1.
L = .1
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

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

function f(x::Float64, y::Float64)
  return -1.
end
  
function uexact(xx::Float64, yy::Float64)
  return .734
end

println()
space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")


#
# Assembly of constant preconditioners
#
g_0 = zeros(Float64, nnode)

printlnln("do_isotropic_elliptic_assembly for ξ = 0 ...")
A_0, _ = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(g_0), f, uexact)

printlnln("assemble amg_0 preconditioner for A_0 ...")
Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);

printlnln("prepare_lorasc_precond for A_0 ...")
Π_lorasc_0 = @time prepare_lorasc_precond(tentative_nnode,
                                          ndom,
                                          cells,
                                          points,
                                          cell_neighbors,
                                          exp.(g_0),
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          f,
                                          uexact)


#
# Load kl representation and prepare mcmc sampler
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
                                              
#mcmc_sampler = prepare_mcmc_sampler(Λ, Ψ)
mc_sampler = prepare_mc_sampler(Λ, Ψ)


println("do_isotropic_elliptic_assembly for ξ_1 ...")
A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(g_0), f, uexact)


#
# Sample ξ by mcmc and solve linear systems by deflated pcg with 
# online eigenvector approximations
#
nsmp = 1_000
cnt_reals = ones(Int, nsmp)
using Statistics: var

for s in 2:nsmp

  println("\n$s / $nsmp")
  #cnt_reals[s] = cnt_reals[s-1] + draw!(mcmc_sampler)
  draw!(mc_sampler)

  print("do_isotropic_elliptic_assembly ... ")  
  @time update_isotropic_elliptic_assembly!(A, b,
                                            cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            exp.(mc_sampler.g),
                                            f, uexact)

  print("amg_0-pcg of A * u = b ... ")
  u, it, _ = @time pcg(A, b, M=Π_amg_0)
  space_println("iter = $it")
                                         
  print("lorasc_0-pcg solve of A * u = b ...")
  u, it, _ = @time pcg(A, b, M=Π_lorasc_0)
  space_println("iter = $it")

end                                          


printlnln("ld-def-amg_0-pcg solve of A * u = b ...")
u, it, _ = @time defpcg(A, b, ϕ_ld, M=Π_amg_0);
space_println("n = $(A.n), nev = $nev (ld), iter = $it")
                                         
printlnln("ld-def-lorasc_0-pcg solve of A * u = b ...")
u, it, _ = @time defpcg(A, b, ϕ_ld, M=Π_lorasc_0);
space_println("n = $(A.n), nev = $nev (ld), iter = $it")