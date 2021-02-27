push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./MyPreconditioners/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: cg, pcg, defpcg, eigpcg, eigdefpcg
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using MyPreconditioners: BJPreconditioner, BJop,
                         Cholesky16, get_cholesky16,
                         Cholesky32, get_cholesky32

using Utils: space_println, printlnln, 
             save_deflated_system, save_system,
             load_system 

using NPZ: npzread, npzwrite
using Random: seed!
using LinearMaps: LinearMap
import SuiteSparse
import JLD

using LinearAlgebra: isposdef, rank

troubleshoot = true

maxit = 5_000
tentative_nnode = 20_000
load_existing_mesh = false

ndom = 8
load_existing_partition = false

nbj = ndom

nvec = floor(Int, 1.25 * ndom)
spdim = 3 * ndom

const preconds_with_dd = ("lorasc$(ndom)_0", "lorasc$(ndom)_1", "neumann-neumann$(ndom)_0")

preconds = ["neumann-neumann$(ndom)_0"] # ∈ ("amg_0", "bj$(nbj)_0") ⋃ preconds_with_dd 

nchains = 50
nsmp = 5
seed!(481_456)

model = "SExp"
sig2 = 1.
L = .1

root_fname = get_root_filename(model, sig2, L, tentative_nnode)

function f(x::Float64, y::Float64)
  return -1.
end
  
function uexact(xx::Float64, yy::Float64)
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
# Load mesh parition
#
do_dd = false
for precond in preconds
  if precond in preconds_with_dd
    global do_dd = true
    break
  end
end

if do_dd
  if load_existing_partition
    epart, npart = load_partition(tentative_nnode, ndom)
  else
    epart, npart = mesh_partition(cells, ndom)
    save_partition(epart, npart, tentative_nnode, ndom)
  end

  ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l, node_owner,
  elemd, node_Γ, node_Γ_cnt, node_Id, nnode_Id = set_subdomains(cells,
                                                                cell_neighbors,
                                                                epart, 
                                                                npart,
                                                                dirichlet_inds_g2l)
end

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


#
# Load kl representation and prepare mcmc sampler
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")


iters = test_solvers_on_several_chains(nchains, nsmp, Λ, Ψ, Π,
                                       preconds, nvec, spdim, maxit,
                                       do_pcg=true,
                                       do_eigpcg=false,
                                       do_eigdefpcg=false,
                                       do_defpcg=false,
                                       save_results=false,
                                       troubleshoot=troubleshoot)
