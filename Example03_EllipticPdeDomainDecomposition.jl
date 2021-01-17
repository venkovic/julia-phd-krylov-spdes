push!(LOAD_PATH, "./Fem/")
import Pkg
Pkg.activate(".")
using Fem

using LinearMaps
using IterativeSolvers
using Preconditioners

tentative_nnode = 100_000
load_existing_mesh = true

ndom = 400
load_existing_partition = true

if load_existing_mesh
  cells, points, point_markers, cell_neighbors = load_mesh(tentative_nnode)
else
  mesh = get_mesh(tentative_nnode)
  save_mesh(mesh, tentative_nnode)
  cells = mesh.cell
  points = mesh.point
  point_markers = mesh.point_marker
  cell_neighbors = mesh.cell_neighbor
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

ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l, node_owner,
elemd, node_Γ, node_Γ_cnt, node_Id, nnode_Id = set_subdomains(cells,
                                                              cell_neighbors,
                                                              epart, 
                                                              npart,
                                                              dirichlet_inds_g2l)

function a(x::Float64, y::Float64)
  return .1 + .0001 * x * y
end
  
function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx::Float64, yy::Float64)
  return .734
end

println("nnode = $(size(points)[2])")
println("nel = $(size(cells)[2])")

print("do_isotropic_elliptic_assembly ...")
A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            a, f, uexact)

print("prepare_global_schur ...")
A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ = @time prepare_global_schur(cells,
                                                           points,
                                                           epart,
                                                           ind_Id_g2l,
                                                           ind_Γ_g2l,
                                                           node_owner,
                                                           a,
                                                           f,
                                                           uexact)


print("assemble amg preconditioners of A_IId ...")
Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IId[idom])
               for idom in 1:ndom];

n_Γ, _ = size(A_ΓΓ)
S = LinearMap(x -> apply_global_schur(A_IId, A_IΓd, A_ΓΓ, x, preconds=Π_IId), n_Γ, issymmetric=true)

print("get_schur_rhs ...")
b_schur = @time get_schur_rhs(b_Id, A_IId, A_IΓd, b_Γ)

print("solve for u_Γ ...")
u_Γ = @time IterativeSolvers.cg(S, b_schur)

print("get_subdomain_solutions ...")
u_Id = @time get_subdomain_solutions(u_Γ, A_IId, A_IΓd, b_Id)

u_with_dd = merge_subdomain_solutions(u_Γ, u_Id, node_Γ, node_Id,
                                      dirichlet_inds_l2g, uexact,
                                      points)

print("assemble amg preconditioner of A ...")
Π = @time AMGPreconditioner{SmoothedAggregation}(A)

print("solve for u_no_dd_no_dirichlet ...")
u_no_dd_no_dirichlet = @time IterativeSolvers.cg(A, b, Pl=Π)

u_no_dd = append_bc(dirichlet_inds_l2g, not_dirichlet_inds_l2g,
                    u_no_dd_no_dirichlet, points, uexact)

println("extrema(u_with_dd - u_no_dd) = $(extrema(u_with_dd - u_no_dd))")


print("prepare_local_schurs ...")
A_IIdd, A_IΓdd, A_ΓΓdd, _, _ = @time prepare_local_schurs(cells,
                                                          points,
                                                          epart,
                                                          ind_Id_g2l,
                                                          ind_Γd_g2l,
                                                          ind_Γ_g2l,
                                                          node_owner,
                                                          a,
                                                          f,
                                                          uexact)

ΠSnn = prepare_neumann_neumann_schur_precond(A_IIdd,
                                             A_IΓdd,
                                             A_ΓΓdd,
                                             ind_Γd_Γ2l,
                                             node_Γ_cnt,
                                             preconds=Π_IId)

S2 = LinearMap(x -> apply_local_schurs(A_IIdd,
                                       A_IΓdd,
                                       A_ΓΓdd,
                                       ind_Γd_Γ2l,
                                       node_Γ_cnt,
                                       x,
                                       preconds=Π_IId),
                                       n_Γ, issymmetric=true)

Sd = [LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                        A_IΓdd[idom],
                                        A_ΓΓdd[idom],
                                        xd,
                                        precond=Π_IId[idom]),
                                        ind_Γd_g2l[idom].count, issymmetric=true)
                                        for idom in 1:ndom]

using LinearAlgebra
isposdef(A)
isposdef(S)

println("extrema(S * b_schur - S2 * b_schur) = $(extrema(S * b_schur - S2 * b_schur))")

print("solve for u_Γ with neumann-neumann preconditioner ...")
u_Γ = @time IterativeSolvers.cg(S, b_schur, Pl=ΠSnn)

A_ΓΓ2 = assemble_A_ΓΓ_from_local_blocks(A_ΓΓdd, ind_Γd_Γ2l)
println(extrema(A_ΓΓ - A_ΓΓ2))