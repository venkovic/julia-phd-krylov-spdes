push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./Utils/")
push!(LOAD_PATH, "./MyPreconditioners/")
import Pkg
Pkg.activate(".")
using Fem
using RecyclingKrylovSolvers: cg, pcg, defpcg

using Utils: space_println, printlnln

using MyPreconditioners: BJPreconditioner
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using NPZ: npzread, npzwrite
using Random: seed!; seed!(123_456);
using LinearMaps: LinearMap
using SparseArrays: SparseMatrixCSC
import Arpack

tentative_nnode = 4_000
load_existing_mesh = false
save_spectra = true

ndom = 20 # 5, 10, 20, 30, 80, 200
load_existing_partition = false

nreals = 3

model = "SExp"
sig2 = 1.
L = 1. # .1, 1.
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
gs = [copy(g) for _ in 1:nreals]
for ireal in 1:nreals
  @time draw!(Λ, Ψ, ξ, g)
  gs[ireal] .= g
end

printlnln("do_isotropic_elliptic_assembly for ξ ...")
As, bs = [], []
for ireal in 1:nreals
  A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(gs[ireal]), f, uexact)
  push!(As, A)
  push!(bs, b)  
end
                                            
#
# Preconditioner assemblies
#
printlnln("assemble amg_0 preconditioner for A_0 ...")
Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);

Π_amg_ts = []
for ireal in 1:nreals
  printlnln("assemble amg_t preconditioner for A_$ireal ...")
  Π_amg_t = @time AMGPreconditioner{SmoothedAggregation}(As[ireal]);
  push!(Π_amg_ts, Π_amg_t)
end

printlnln("prepare_lorasc_precond ...")
Π_lorasc_0 = @time prepare_lorasc_precond(S_0,
                                          A_IId_0,
                                          A_IΓd_0,
                                          A_ΓΓ_0,
                                          ind_Id_g2l,
                                          ind_Γ_g2l,
                                          not_dirichlet_inds_g2l)

printlnln("prepare_lorasc_precond with ε = 0.01 ...")
Π_lorasc_ε01_0 = @time prepare_lorasc_precond(tentative_nnode,
                                              ndom,
                                              cells,
                                              points,
                                              cell_neighbors,
                                              exp.(g_0),
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              f,
                                              uexact,
                                              ε=.01)


printlnln("prepare_lorasc_precond with ε = 0 ...")
Π_lorasc_ε00_0 = @time prepare_lorasc_precond(tentative_nnode,
                                              ndom,
                                              cells,
                                              points,
                                              cell_neighbors,
                                              exp.(g_0),
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              f,
                                              uexact,
                                              ε=0)

Π_lorasc_ε01_ts, Π_lorasc_ε00_ts = [], []
for ireal in 1:nreals
  printlnln("assemble lorasc preconditioner with ε = 0.01 for A_$ireal ...")
  Π_lorasc_ε01_t = @time prepare_lorasc_precond(tentative_nnode,
                                                ndom,
                                                cells,
                                                points,
                                                cell_neighbors,
                                                exp.(gs[ireal]),
                                                dirichlet_inds_g2l,
                                                not_dirichlet_inds_g2l,
                                                f,
                                                uexact,
                                                ε=.01)
  push!(Π_lorasc_ε01_ts, Π_lorasc_ε01_t)
end
for ireal in 1:nreals
  printlnln("assemble lorasc preconditioner with ε = 0 for A_$ireal ...")
  Π_lorasc_ε00_t = @time prepare_lorasc_precond(tentative_nnode,
                                                ndom,
                                                cells,
                                                points,
                                                cell_neighbors,
                                                exp.(gs[ireal]),
                                                dirichlet_inds_g2l,
                                                not_dirichlet_inds_g2l,
                                                f,
                                                uexact,
                                                ε=0)
  push!(Π_lorasc_ε00_ts, Π_lorasc_ε00_t)
end

Π_bJ_0 = BJPreconditioner(ndom, A_0)

Π_bJ_ts = []
for ireal in 1:nreals
  Π_bJ_t = BJPreconditioner(ndom, As[ireal])
  push!(Π_bJ_ts, Π_bJ_t)
end


#
# Solve for some eigenpairs
#
function apply_preconditioner_get_eigenpairs(Π, tag)
  λ_lds, Φ_lds = [], []
  for ireal in 1:nreals
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
    Π_inv_At = Array{Float64}(undef, As[ireal].n, As[ireal].n)
    @time for j in 1:As[ireal].n
      if size(Π) == (1,)
        Π_inv_At[:, j] .= Π[1] \ Array(As[ireal][:, j])
      else
        Π_inv_At[:, j] .= Π[ireal] \ Array(As[ireal][:, j])
      end
    end
    λ_ld, Φ_ld = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
    push!(λ_lds, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
    push!(Φ_lds, Φ_ld)
  end        
  λ_mds, Φ_mds = [], []                                    
  for ireal in 1:nreals
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
    Π_inv_At = Array{Float64}(undef, As[ireal].n, As[ireal].n)
    @time for j in 1:As[ireal].n
      if size(Π) == (1,)
        Π_inv_At[:, j] .= Π[1] \ Array(As[ireal][:, j])
      else
        Π_inv_At[:, j] .= Π[ireal] \ Array(As[ireal][:, j])
      end
    end
    λ_md, Φ_md = @time Arpack.eigs(Π_inv_At, nev=As[ireal].n-100, which=:LM)
    push!(λ_mds, λ_md)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)
    push!(Φ_mds, Φ_md)
  end
end

if save_spectra
  #apply_preconditioner_get_eigenpairs([Π_amg_0], "amg_0")
  #apply_preconditioner_get_eigenpairs(Π_amg_ts, "amg_t")
  apply_preconditioner_get_eigenpairs([Π_lorasc_ε00_0], "lorasc_ndom$ndom"*"_eps00_0")
  apply_preconditioner_get_eigenpairs(Π_lorasc_ε00_ts, "lorasc_ndom$ndom"*"_eps00_t")
  apply_preconditioner_get_eigenpairs([Π_lorasc_ε01_0], "lorasc_ndom$ndom"*"_eps01_0")
  apply_preconditioner_get_eigenpairs(Π_lorasc_ε01_ts, "lorasc_ndom$ndom"*"_eps01_t")
  apply_preconditioner_get_eigenpairs([Π_bJ_0], "bJ_nb$ndom"*"_0")
  apply_preconditioner_get_eigenpairs(Π_bJ_ts, "bJ_nb$ndom"*"_t")

  #conds_Πinv_A
end

if save_spectra

else
end







#
# Preconditioner applications
#
println()

print("apply amg_0 ...")
@time Π_amg_0 \ rand(As[1].n);

for ireal in 1:nreals
  print("apply amg_t_$ireal ...")
  @time Π_amg_ts[ireal] \ rand(As[ireal].n);
end

print("apply lorasc_0 ...")
@time Π_lorasc_0 \ rand(As[1].n)


#
# Pcg solves
#
for ireal in 1:nreals
  printlnln("amg_t-pcg of A * u = b ...")
  u, it, _ = @time pcg(As[ireal], bs[ireal], zeros(As[ireal].n), Π_amg_ts[ireal])
  space_println("n = $(As[ireal].n), iter = $it")

  printlnln("amg_0-pcg of A * u = b ...")
  u, it, _ = @time pcg(As[ireal], bs[ireal], zeros(As[ireal].n), Π_amg_0)
  space_println("n = $(As[ireal].n), iter = $it")

  printlnln("cg solve of A * u = b ...")
  u, it, _ = @time cg(As[ireal], bs[ireal], zeros(As[ireal].n))
  space_println("n = $(As[ireal].n), iter = $it")

  """
  #printlnln("ld-def-amg_0-pcg solve of A * u = b ...")
  u, it, _ = @time defpcg(As[ireal], bs[ireal], zeros(As[ireal].n), Φ_ld, Π_amg_0);
  space_println("n = $(As[ireal].n), nev = $nev (ld), iter = $it")

  printlnln("md-def-amg_0-pcg solve of A * u = b ...")
  u, it, _ = @time defpcg(As[ireal], bs[ireal], zeros(As[ireal].n), Φ_md, Π_amg_0);
  space_println("n = $(As[ireal].n), nev = $nev (md), iter = $it")
  """

  printlnln("lorasc_0-pcg solve of A * u = b ...")
  u, it, _ = @time pcg(As[ireal], bs[ireal], zeros(As[ireal].n), Π_lorasc_0)
  space_println("n = $(As[ireal].n), iter = $it")

  """
  printlnln("ld-def-lorasc_0-pcg solve of A * u = b ...")
  u, it, _ = @time defpcg(As[ireal], bs[ireal], zeros(As[ireal].n), Φ_ld, Π_lorasc_0);
  space_println("n = $(As[ireal].n), nev = $nev (ld), iter = $it")
  """
end