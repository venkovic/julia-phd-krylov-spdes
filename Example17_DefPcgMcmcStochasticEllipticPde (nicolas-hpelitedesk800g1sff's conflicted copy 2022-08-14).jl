import Pkg
Pkg.activate(".")


push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./MyPreconditioners/")
push!(LOAD_PATH, "./Utils/")


using Distributed
using Fem
using RecyclingKrylovSolvers: pcg
using RecyclingKrylovSolvers: rrdefpcg, rrpcg, hrdefpcg, hrpcg
using RecyclingKrylovSolvers: trrrdefpcg, trrrpcg, trhrdefpcg, trhrpcg
using RecyclingKrylovSolvers: lotrrrdefpcg, lotrrrpcg, lotrhrdefpcg, lotrhrpcg
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using MyPreconditioners: BJPreconditioner
using Utils: space_println, printlnln, 
             save_deflated_system, save_system,
             load_system 
using NPZ: npzread, npzwrite
using Random: seed!
using LinearMaps: LinearMap
import SuiteSparse
import JLD
using LinearAlgebra: isposdef, rank
using ParallelDataTransfer

include("Example17_DefPcgMcmcStochasticEllipticPde_Functions.jl")

maxit = 4_000
tentative_nnode = 128_000 # 4_000, 8_000, 16_000, 32_000, 64_000, 128_000 @ 10 subdomains
load_existing_mesh = false
ndom = 10 # 5, 7, 10, 15, 20, 30 @ 32_000 DoFs
load_existing_partition = false
nbj = ndom
nvec = ndom
spdim = 3 * ndom
precond = "bj$(nbj)_0"  # ∈ ("bj$(nbj)_0",
                        #    "lorasc$(ndom)_1")

const preconds_with_dd = ("lorasc$(ndom)_0",
                          "lorasc$(ndom)_1",
                          "neumann-neumann$(ndom)_0")

nchains = 100
nsmp = 10
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
# Load mesh parition
#
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
# Load kl representation
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")

#
# Prepare constant preconditioners
#
Π = get_constant_precond(precond, A_0)
#
# Run test
#
iters = test_solvers_on_several_chains(nchains, nsmp, Λ, Ψ, Π,
                                       precond, nvec, spdim, maxit)
