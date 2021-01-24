push!(LOAD_PATH, "./Utils/")
push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
import Pkg
Pkg.activate(".")
using Fem
using RecyclingKrylovSolvers

using Utils: space_println, printlnln

using LinearMaps
using Preconditioners

tentative_nnode = 200_000
load_existing_mesh = false

ndom = 50
load_existing_partition = false

# Remarks:
# - Use assemble_local_schurs() for a faster set-up and application
#   of the NeumannNeumannSchurPreconditioner
# - NeumannNeumannSchurPreconditioner-pcg performs worse for larger ndom
# - def-NeumannNeumannSchurPreconditioner-pcg works very well with least
#   dominant (ld) eigenvectors

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

space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

printlnln("do_isotropic_elliptic_assembly ...")
A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            a, f, uexact)

printlnln("prepare_global_schur ...")
A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ = @time prepare_global_schur(cells,
                                                           points,
                                                           epart,
                                                           ind_Id_g2l,
                                                           ind_Γ_g2l,
                                                           node_owner,
                                                           a,
                                                           f,
                                                           uexact)

printlnln("assemble amg preconditioners of A_IId ...")
Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IId[idom])
               for idom in 1:ndom];

n_Γ, _ = size(A_ΓΓ)
S_global = LinearMap(x -> apply_global_schur(A_IId, A_IΓd, A_ΓΓ, x, preconds=Π_IId), n_Γ, issymmetric=true)

printlnln("get_schur_rhs ...")
b_schur = @time get_schur_rhs(b_Id, A_IId, A_IΓd, b_Γ, preconds=Π_IId)

printlnln("assemble amg preconditioner of A ...")
Π = @time AMGPreconditioner{SmoothedAggregation}(A)

printlnln("amg-pcg solve of A * u_no_dd_no_dirichlet = b ...")
u_no_dd_no_dirichlet, it, _ = @time pcg(A, b[:, 1], M=Π)
space_println("n = $(A.n), iter = $it")

u_no_dd = append_bc(dirichlet_inds_l2g, not_dirichlet_inds_l2g,
                    u_no_dd_no_dirichlet, points, uexact)

printlnln("prepare_local_schurs ...")
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

printlnln("assemble_local_schurs ...")
Sd_local_mat = @time assemble_local_schurs(A_IIdd, A_IΓdd, A_ΓΓdd, preconds=Π_IId)
                                                          
S_local_mat = LinearMap(x -> apply_local_schurs(Sd_local_mat,
                                                ind_Γd_Γ2l,
                                                node_Γ_cnt,
                                                x),
                                                n_Γ, issymmetric=true)

printlnln("prepare_neumann_neumann_induced_precond using S_local_mat ...")
ΠInducednn_local_mat = @time prepare_neumann_neumann_induced_precond(A_IIdd,
                                                                     A_IΓdd,
                                                                     A_ΓΓdd,
                                                                     ind_Id_g2l,
                                                                     ind_Γ_g2l,
                                                                     ind_Γd_Γ2l,
                                                                     node_Γ_cnt,
                                                                     node_Γ,
                                                                     not_dirichlet_inds_g2l,
                                                                     preconds=Π_IId)





printlnln("prepare_neumann_neumann_schur_precond using S_local_mat ...")
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
#printlnln("prepare_neumann_neumann_schur_precond with local amg-pcg solves ...")
#ΠSnn_local = @time prepare_neumann_neumann_schur_precond(A_IIdd,
#                                                         A_IΓdd,
#                                                         A_ΓΓdd,
#                                                         ind_Γd_Γ2l,
#                                                         node_Γ_cnt,
#                                                         preconds=Π_IId)

printlnln("define (singular) local Schur operators ...")
Sd = @time [LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                              A_IΓdd[idom],
                                              A_ΓΓdd[idom],
                                              xd,
                                              precond=Π_IId[idom]),
                                              ind_Γd_g2l[idom].count, issymmetric=true)
                                              for idom in 1:ndom]

space_println("extrema(S_global * b_schur - S_local_mat * b_schur) = $(extrema(S_global * b_schur - S_local_mat * b_schur))")

printlnln("S * b_schur ...")
@time S_global * b_schur;
printlnln("S_local * b_schur ...")
@time S_local * b_schur;
printlnln("S_local_mat * b_schur ...")
@time S_local_mat * b_schur;

# Kind of slow ...
#printlnln("cg solve of u_Γ s.t. S_global * u_Γ = b_schur ...")
#u_Γ__global, it, _ = @time cg(S_global, b_schur)
#println("n = $(S_global.N), iter = $it")

printlnln("neumann-neumann-pcg solve of S_global * u_Γ = b_schur ...")
u_Γ, it, _ = @time pcg(S_local_mat, b_schur, M=ΠSnn_local_mat);
space_println("n = $(S_local_mat.N), iter = $it")

printlnln("get_subdomain_solutions ...")
u_Id = @time get_subdomain_solutions(u_Γ, A_IId, A_IΓd, b_Id);

printlnln("merge_subdomain_solutions ...")
u_with_dd = @time merge_subdomain_solutions(u_Γ, u_Id, node_Γ, node_Id,
                                            dirichlet_inds_l2g, uexact,
                                            points);

space_println("extrema(u_with_dd - u_no_dd) = $(extrema(u_with_dd - u_no_dd))")

# There's gotta be a betta way!
using SparseArrays
printlnln("assemble global schur ...")
S_sp = @time sparse(S_local_mat)

nev = ndom + 10

using Arpack
printlnln("solve for least dominant eigvecs of schur complement ...")
λ, ϕ = @time Arpack.eigs(S_sp, nev=nev, which=:SM)
printlnln("ld-def-neumann-neumann-pcg solve of S_global * u_Γ = b_schur ...")
u_Γ, it, _ = @time defpcg(S_local_mat, b_schur, ϕ, M=ΠSnn_local_mat);
space_println("n = $(S_local_mat.N), ndom = $ndom, nev = $nev (ld), iter = $it")

printlnln("solve for most dominant eigvecs of schur complement ...")
λ, ϕ = @time Arpack.eigs(S_sp, nev=nev, which=:LM)
printlnln("md-def-neumann-neumann-pcg solve of S_global * u_Γ = b_schur ...")
u_Γ, it, _ = @time defpcg(S_local_mat, b_schur, ϕ, M=ΠSnn_local_mat);
space_println("n = $(S_local_mat.N), ndom = $ndom, nev = $nev (md), iter = $it")


#printlnln("cg solve of A * u = b ...")
#u_no_dd_no_dirichlet, it, _ = @time cg(A, b[:, 1])
#space_println("n = $(A.n), iter = $it")

#printlnln("solve for least dominant eigvecs of A ...")
#λ, ϕ = @time Arpack.eigs(A, nev=nev, which=:SM)
#printlnln("ld-def-neumann-neumann-induced-pcg solve of A * u = b ...")
#u_no_ddno_dirichlet, it, _ = @time defpcg(A, b[:, 1], ϕ, M=ΠInducednn_local_mat);
#space_println("n = $(A.n), ndom = $ndom, nev = $nev (md), iter = $it")

#printlnln("neumann-neumann-induced-pcg solve of A * u = b ...")
#u_no_dd_no_dirichlet, it, _ = @time pcg(A, b[:, 1], M=ΠInducednn_local_mat)
#space_println("n = $(A.n), ndom = $ndom, iter = $it")