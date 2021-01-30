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
import Arpack

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


println()
space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")


M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
g = npzread("data/$root_fname.kl-eigvecs.npz")
ξ, g = draw(Λ, Ψ)

# The eigenfunctions obtained by domain 
# decomposition are not perfectly orthogonal
χ = get_kl_coordinates(g, Λ, Ψ, M)  
printlnln("extrema(ξ - χ) = $(extrema(ξ - χ))")


#
# Operator assemblies for ξ = 0
#
g_0 = copy(g); g_0 .= 0.

printlnln("do_isotropic_elliptic_assembly for ξ = 0 ...")
A_0, _ = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(g_0), f, uexact)

printlnln("prepare_global_schur ...")
A_IId_0, A_IΓd_0, A_ΓΓ_0, b_Id_0, b_Γ_0 = @time prepare_global_schur(cells,
                                                                     points,
                                                                     epart,
                                                                     ind_Id_g2l,
                                                                     ind_Γ_g2l,
                                                                     node_owner,
                                                                     exp.(g_0),
                                                                     f,
                                                                     uexact)
                                             
printlnln("assemble amg preconditioners of A_IId_0 ...")
Π_IId_0 = @time [AMGPreconditioner{SmoothedAggregation}(A_IId_0[idom])
                 for idom in 1:ndom];
                                             
printlnln("prepare_local_schurs ...")
A_IIdd_0, A_IΓdd_0, A_ΓΓdd_0, _, _ = @time prepare_local_schurs(cells,
                                                                points,
                                                                epart,
                                                                ind_Id_g2l,
                                                                ind_Γd_g2l,
                                                                ind_Γ_g2l,
                                                                node_owner,
                                                                exp.(g_0),
                                                                f,
                                                                uexact)

# 
# (Slow-ish) assembly of local Schur complements
#

printlnln("assemble_local_schurs ...")
Sd_local_mat_0 = @time assemble_local_schurs(A_IIdd_0, A_IΓdd_0, A_ΓΓdd_0, preconds=Π_IId_0)
                                            
printlnln("build LinearMap using assembled local schurs ...")
S_0 = LinearMap(x -> apply_local_schurs(Sd_local_mat_0,
                                        ind_Γd_Γ2l,
                                        node_Γ_cnt,
                                        x), nothing,
                                        n_Γ, issymmetric=true)

"""
printlnln("build LinearMap using (p)cg solves ...")
S_0 = LinearMap(x -> apply_local_schurs(A_IIdd_0,
                                        A_IΓdd_0,
                                        A_ΓΓdd_0,
                                        ind_Γd_Γ2l,
                                        node_Γ_cnt,
                                        x,
                                        preconds=Π_IId_0), 
                                        nothing, n_Γ, issymmetric=true)
"""

#
# Operator assemblies for a random ξ_t
#
printlnln("in-place draw of ξ ...")
@time draw!(Λ, Ψ, ξ, g)

printlnln("do_isotropic_elliptic_assembly for ξ ...")
A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            exp.(g), f, uexact)

                                            
#
# Solve for some eigenpairs
#
nev = ndom + 10
printlnln("solve for $nev least dominant eigvecs of A ...")
λ_ld, ϕ_ld = @time Arpack.eigs(A, nev=nev, which=:SM)
                                            
printlnln("solve for $nev most dominant eigvecs of A ...")
λ_md, ϕ_md = @time Arpack.eigs(A, nev=nev, which=:LM)


#
# Preconditioner assemblies
#
printlnln("assemble amg_0 preconditioner for A_0 ...")
Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);

printlnln("assemble amg_t preconditioner for A ...")
Π_amg_t = @time AMGPreconditioner{SmoothedAggregation}(A);

printlnln("prepare_lorasc_precond ...")
Π_lorasc_0 = @time prepare_lorasc_precond(S_0,
                                          A_IId_0,
                                          A_IΓd_0,
                                          A_ΓΓ_0,
                                          ind_Id_g2l,
                                          ind_Γ_g2l,
                                          not_dirichlet_inds_g2l)

printlnln("prepare_lorasc_precond ...")
Π_lorasc_1 = @time prepare_lorasc_precond(tentative_nnode,
                                          ndom,
                                          cells,
                                          points,
                                          cell_neighbors,
                                          exp.(g),
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          f,
                                          uexact)

#
# Preconditioner applications
#
println()

Π_amg_0 \ rand(A.n);
print("apply amg_0 ...")
@time Π_amg_0 \ rand(A.n);

Π_amg_t \ rand(A.n);
print("apply amg_t ...")
@time Π_amg_t \ rand(A.n);

Π_lorasc_0 \ rand(A.n)
print("apply lorasc_0 ...")
@time Π_lorasc_0 \ rand(A.n)


#
# Pcg solves
#
printlnln("amg_t-pcg of A * u = b ...")
u, it, _ = @time pcg(A, b, M=Π_amg_t)
space_println("n = $(A.n), iter = $it")

printlnln("amg_0-pcg of A * u = b ...")
u, it, _ = @time pcg(A, b, M=Π_amg_0)
space_println("n = $(A.n), iter = $it")

printlnln("cg solve of A * u = b ...")
u, it, _ = @time cg(A, b)
space_println("n = $(A.n), iter = $it")

printlnln("ld-def-amg_0-pcg solve of A * u = b ...")
u, it, _ = @time defpcg(A, b, ϕ_ld, M=Π_amg_0);
space_println("n = $(A.n), nev = $nev (ld), iter = $it")

printlnln("md-def-amg_0-pcg solve of A * u = b ...")
u, it, _ = @time defpcg(A, b, ϕ_md, M=Π_amg_0);
space_println("n = $(A.n), nev = $nev (md), iter = $it")

printlnln("lorasc_0-pcg solve of A * u = b ...")
u, it, _ = @time pcg(A, b, M=Π_lorasc_0)
space_println("n = $(A.n), iter = $it")
                                         
printlnln("ld-def-lorasc_0-pcg solve of A * u = b ...")
u, it, _ = @time defpcg(A, b, ϕ_ld, M=Π_lorasc_0);
space_println("n = $(A.n), nev = $nev (ld), iter = $it")
