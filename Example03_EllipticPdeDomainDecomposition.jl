#using TriangleMesh
#using NPZ
push!(LOAD_PATH, "./Fem/")

import Pkg
Pkg.activate(".")

using Fem
using LinearMaps
using IterativeSolvers
using Preconditioners

tentative_nnode = 100_000
mesh = get_mesh(tentative_nnode)

dirichlet_inds_g2l, not_dirichlet_inds_g2l,
dirichlet_inds_l2g, not_dirichlet_inds_l2g = 
get_dirichlet_inds(mesh.point, mesh.point_marker)

ndom = 400
epart, npart = mesh_partition(mesh, ndom)
ind_Id_g2l, ind_Γ_g2l, node_owner, 
elemd, node_Γ, node_Id, nnode_Id = set_subdomains(mesh, epart, npart, dirichlet_inds_g2l)
# only ind_Id_g2l, ind_Γ_g2l and node_owner are essential

function a(x::Float64, y::Float64)
  return .1 + .0001 * x * y
end
  
function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx::Float64, yy::Float64)
  return .734
end

print("do_isotropic_elliptic_assembly ...")
A, b = @time do_isotropic_elliptic_assembly(mesh.cell, mesh.point,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            mesh.point_marker,
                                            a, f, uexact)

print("do_schur_assembly ...")
A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ = @time do_schur_assembly(mesh.cell,
                                                       mesh.point,
                                                       epart,
                                                       ind_Id_g2l,
                                                       ind_Γ_g2l,
                                                       node_owner,
                                                       a,
                                                       f,
                                                       uexact)


print("assemble AMG preconditioners ...")
Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IId[idom])
               for idom in 1:ndom];

n_Γ, _ = size(A_ΓΓ)
#S = LinearMap(x -> apply_schur(A_IId, A_IΓd, A_ΓΓ, x), n_Γ, issymmetric=true)
S = LinearMap(x -> apply_schur(A_IId, A_IΓd, A_ΓΓ, x, Π_IId), n_Γ, issymmetric=true)
print("get_schur_rhs ...")
b_schur = @time get_schur_rhs(b_Id, A_IId, A_IΓd, b_Γ)

print("solve for u_Γ ...")
u_Γ = @time IterativeSolvers.cg(S, b_schur)
print("get_subdomain_solutions ...")
u_Id = @time get_subdomain_solutions(u_Γ, A_IId, A_IΓd, b_Id)

u_with_dd = merge_subdomain_solutions(u_Γ, u_Id, node_Γ, node_Id,
                                      dirichlet_inds_l2g, uexact,
                                      mesh.point)

print("solve for u_no_dd_no_dirichlet ...")
u_no_dd_no_dirichlet = @time IterativeSolvers.cg(A, b)

u_no_dd = append_bc(dirichlet_inds_l2g, not_dirichlet_inds_l2g,
                    u_no_dd_no_dirichlet, mesh.point, uexact)

print("extrema(u_with_dd - u_no_dd) = $(extrema(u_with_dd - u_no_dd))")





