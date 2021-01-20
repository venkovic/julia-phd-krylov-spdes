push!(LOAD_PATH, "./Fem/")
import Pkg
Pkg.activate(".")
using Fem

using LinearMaps
using IterativeSolvers
using Preconditioners

tentative_nnode = 200_000
load_existing_mesh = false

ndom = 50
load_existing_partition = false

# Remarks:
# - NeumannNeumannSchurPreconditioner performs worse for larger ndom
# - More CPU time is needed to apply the Schur complement (without
#   parallelization) than to apply A

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
S_global = LinearMap(x -> apply_global_schur(A_IId, A_IΓd, A_ΓΓ, x, preconds=Π_IId), n_Γ, issymmetric=true)

print("get_schur_rhs ...")
b_schur = @time get_schur_rhs(b_Id, A_IId, A_IΓd, b_Γ)

print("assemble amg preconditioner of A ...")
Π = @time AMGPreconditioner{SmoothedAggregation}(A)

print("amg-pcg solve of u_no_dd_no_dirichlet s.t. A * u_no_dd_no_dirichlet = b ...")
u_no_dd_no_dirichlet = @time IterativeSolvers.cg(A, b, Pl=Π)

u_no_dd = append_bc(dirichlet_inds_l2g, not_dirichlet_inds_l2g,
                    u_no_dd_no_dirichlet, points, uexact)

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

print("assemble_local_schurs ...")
Sd_local_mat = @time assemble_local_schurs(A_IIdd, A_IΓdd, A_ΓΓdd, preconds=Π_IId)
                                                          
S_local_mat = LinearMap(x -> apply_local_schurs(Sd_local_mat,
                                                ind_Γd_Γ2l,
                                                node_Γ_cnt,
                                                x),
                                                n_Γ, issymmetric=true)

print("prepare_neumann_neumann_schur_precond using S_local_mat ...")
ΠSnn_local_mat = @time prepare_neumann_neumann_schur_precond(Sd_local_mat,
                                                             ind_Γd_Γ2l,
                                                             node_Γ_cnt)

S_local = LinearMap(x -> apply_local_schurs(A_IIdd,
                                            A_IΓdd,
                                            A_ΓΓdd,
                                            ind_Γd_Γ2l,
                                            node_Γ_cnt,
                                            x,
                                            preconds=Π_IId),
                                            n_Γ, issymmetric=true)

# Kind of slow ...
#print("prepare_neumann_neumann_schur_precond with local amg-pcg solves ...")
#ΠSnn_local = @time prepare_neumann_neumann_schur_precond(A_IIdd,
#                                                         A_IΓdd,
#                                                         A_ΓΓdd,
#                                                         ind_Γd_Γ2l,
#                                                         node_Γ_cnt,
#                                                         preconds=Π_IId)

print("Define (singular) local Schur operators ...")
Sd = @time [LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                              A_IΓdd[idom],
                                              A_ΓΓdd[idom],
                                              xd,
                                              precond=Π_IId[idom]),
                                              ind_Γd_g2l[idom].count, issymmetric=true)
                                              for idom in 1:ndom]

println("extrema(S_global * b_schur - S_local_mat * b_schur) = $(extrema(S_global * b_schur - S_local_mat * b_schur))")

print("S * b_schur ...")
@time S_global * b_schur;
print("S_local * b_schur ...")
@time S_local * b_schur;
print("S_local_mat * b_schur ...")
@time S_local_mat * b_schur;

# Kind of slow ...
#print("cg solve of u_Γ s.t. S_global * u_Γ = b_schur ...")
#u_Γ__global = @time IterativeSolvers.cg(S_global, b_schur, verbose=true)

print("neumann-neumann-pcg solve of u_Γ s.t. S_global * u_Γ = b_schur ...")
u_Γ = @time IterativeSolvers.cg(S_local_mat, b_schur, Pl=ΠSnn_local_mat, verbose=true);

print("get_subdomain_solutions ...")
u_Id = @time get_subdomain_solutions(u_Γ, A_IId, A_IΓd, b_Id);

print("merge_subdomain_solutions ...")
u_with_dd = @time merge_subdomain_solutions(u_Γ, u_Id, node_Γ, node_Id,
                                            dirichlet_inds_l2g, uexact,
                                            points);

println("extrema(u_with_dd - u_no_dd) = $(extrema(u_with_dd - u_no_dd))")