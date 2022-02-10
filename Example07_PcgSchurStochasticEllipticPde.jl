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
using NPZ

model = "SExp"
sig2 = 1.
L = .1

nreals = 1_000

tentative_nnode = 4_000 # 4_000, 8_000, 16_000, 32_000, 64_000, 128_000
load_existing_mesh = true
save_spectra = false
save_conditioning = false
do_amg = true
do_assembly_of_local_schurs = true # true for ndom = 200, false for ndom = 5

ndom = 200 # 5, 10, 20, 30, 80, 200
load_existing_partition = false

root_fname = get_root_filename(model, sig2, L, tentative_nnode)

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

n_Γ = ind_Γ_g2l.count

function f(x::Float64, y::Float64)
  return -1.
end
  
function uexact(xx::Float64, yy::Float64)
  return .734
end

M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
g = npzread("data/$root_fname.kl-eigvecs.npz")
ξ, g = draw(Λ, Ψ)

space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

printlnln("in-place draw ...")
@time draw!(Λ, Ψ, ξ, g)

printlnln("do_isotropic_elliptic_assembly ...")
A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            exp.(g), f, uexact)

printlnln("assemble amg preconditioner of A ...")
Π = @time AMGPreconditioner{SmoothedAggregation}(A)

#printlnln("prepare_global_schur ...")
#A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ = @time prepare_global_schur(cells,
#                                                           points,
#                                                           epart,
#                                                           ind_Id_g2l,
#                                                           ind_Γ_g2l,
#                                                           node_owner,
#                                                           exp.(g),
#                                                           f,
#                                                           uexact)

printlnln("prepare_local_schurs ...")
A_IIdd, A_IΓdd, A_ΓΓdd, b_Idd, b_Γ = @time prepare_local_schurs(cells,
                                                          points,
                                                          epart,
                                                          ind_Id_g2l,
                                                          ind_Γd_g2l,
                                                          ind_Γ_g2l,
                                                          node_owner,
                                                          exp.(g),
                                                          f,
                                                          uexact)

printlnln("assemble amg preconditioners of A_IId ...")
Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IIdd[idom])
               for idom in 1:ndom];

printlnln("get_schur_rhs ...")
b_schur = @time get_schur_rhs(b_Idd, A_IIdd, A_IΓdd, b_Γ, ind_Γd_Γ2l, preconds=Π_IId)

printlnln("assemble_local_schurs ...")
Sd_local_mat = @time assemble_local_schurs(A_IIdd, A_IΓdd, A_ΓΓdd, preconds=Π_IId)
                                                                            
S_local_mat = LinearMap(x -> apply_local_schurs(Sd_local_mat,
                                                ind_Γd_Γ2l,
                                                node_Γ_cnt,
                                                x),
                                                n_Γ, issymmetric=true)

printlnln("prepare_neumann_neumann_schur_precond using S_local_mat ...")
ΠSnn_local_mat = @time prepare_neumann_neumann_schur_precond(Sd_local_mat,
                                                             ind_Γd_Γ2l,
                                                             node_Γ_cnt)

printlnln("neumann-neumann-pcg solve of u_Γ s.t. S_global * u_Γ = b_schur ...")
u_Γ, it, _ = @time pcg(S_local_mat, b_schur, zeros(S_local_mat.N), ΠSnn_local_mat);
space_println("n = $(S_local_mat.N), iter = $it")

printlnln("A_ΓΓ-pcg solve of u_Γ s.t. S_global * u_Γ = b_schur ...")
u_Γ, it, _ = @time pcg(S_local_mat, b_schur, zeros(S_local_mat.N), A_ΓΓdd);
space_println("n = $(S_local_mat.N), iter = $it")
















"""                                                             
printlnln("amg-pcg solve of u_no_dd_no_dirichlet s.t. A * u_no_dd_no_dirichlet = b ...")
u_no_dd_no_dirichlet, it, _ = @time pcg(A, b[:, 1], zeros(A.n), Π)
space_println("n = $(A.n), iter = $it")
u_no_dd = append_bc(dirichlet_inds_l2g, not_dirichlet_inds_l2g,
                    u_no_dd_no_dirichlet, points, uexact)

printlnln("get_subdomain_solutions ...")
#u_Id = @time get_subdomain_solutions(u_Γ, A_IIdd, A_IΓdd, b_Idd);
                                                
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
printlnln("ld-def-neumann-neumann-pcg solve of u_Γ s.t. S_global * u_Γ = b_schur ...")
u_Γ, it, _ = @time defpcg(S_local_mat, b_schur, ϕ, M=ΠSnn_local_mat);
space_println("n = $(S_local_mat.N), ndom = $ndom, nev = $nev (ld), iter = $it")
"""