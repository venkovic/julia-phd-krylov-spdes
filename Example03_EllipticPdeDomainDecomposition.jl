using TriangleMesh
using NPZ
push!(LOAD_PATH, "./Fem/")
using Fem
using LinearMaps
using IterativeSolvers

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



A, b = @time do_isotropic_elliptic_assembly(mesh.cell, mesh.point,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            mesh.point_marker,
                                            a, f, uexact)

A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ = @time do_schur_assembly(mesh.cell,
                                                       mesh.point,
                                                       epart,
                                                       ind_Id_g2l,
                                                       ind_Γ_g2l,
                                                       node_owner,
                                                       a,
                                                       f,
                                                       uexact)

n_Γ, _ = size(A_ΓΓ)
S = LinearMap(x -> apply_schur(A_IId, A_IΓd, A_ΓΓ, x), n_Γ, issymmetric=true)
b_schur = get_schur_rhs(b_Id, A_IId, A_IΓd, b_Γ)

u_Γ = IterativeSolvers.cg(S, b_schur)
u_Id = get_subdomain_solutions(u_Γ, A_IId, A_IΓd, b_Id)

u = merge_subdomain_solutions(u_Γ, u_Id, node_Γ, node_Id,
                              dirichlet_inds_l2g, uexact,
                              mesh.point)

u_ref = IterativeSolvers.cg(A, b)



#@time npzwrite("cells.npz", mesh.cell' .- 1)
#@time npzwrite("points.npz", mesh.point')
#@time npzwrite("epart.npz", epart .- 1)
#@time npzwrite("nodes_at_interface.npz", node_Γ .- 1)
#for id in 1:ndom
#  npzwrite("nodes_inside_$id.npz", node_Id[id] .- 1)
#end

