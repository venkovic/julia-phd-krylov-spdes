push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./Utils/")
push!(LOAD_PATH, "./MyPreconditioners/")
import Pkg
Pkg.activate(".")
using Fem
using RecyclingKrylovSolvers: cg, pcg, defpcg
using ArnoldiMethod

using Utils: space_println, printlnln

using MyPreconditioners: BJPreconditioner
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using NPZ: npzread, npzwrite
using Random: seed!; seed!(123_456);
using LinearMaps: LinearMap
using SparseArrays: SparseMatrixCSC
import Arpack

tentative_nnode = 8_000 # 4_000, 8_000, 16_000, 32_000, 64_000, 128_000
load_existing_mesh = false
save_spectra = false
save_conditioning = false
do_amg = true
do_assembly_of_local_schurs = true # true for ndom = 200, false for ndom = 5

ndom = 200 # 5, 10, 20, 30, 80, 200
load_existing_partition = false

nreals = 1_000

model = "SExp"
sig2 = 1.
L = .1 # .1, 1.
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
# Operator assembly for ξ = 0
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

if do_assembly_of_local_schurs
  printlnln("assemble_local_schurs ...")
  Sd_local_mat_0 = @time assemble_local_schurs(A_IIdd_0, A_IΓdd_0, A_ΓΓdd_0, preconds=Π_IId_0)
                                              
  printlnln("build LinearMap using assembled local schurs ...")
  S_0 = LinearMap(x -> apply_local_schurs(Sd_local_mat_0,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x), nothing,
                                          n_Γ, issymmetric=true)
else
  printlnln("build LinearMap using (p)cg solves ...")
  S_0 = LinearMap(x -> apply_local_schurs(A_IIdd_0,
                                          A_IΓdd_0,
                                          A_ΓΓdd_0,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x,
                                          preconds=Π_IId_0), 
                                          nothing, n_Γ, issymmetric=true)
end

#
# Realization assemblies for a random ξ_t
#
printlnln("in-place draw of ξ ...")
gs = [copy(g) for _ in 1:nreals]
for ireal in 1:nreals
  @time draw!(Λ, Ψ, ξ, g)
  gs[ireal] .= g
end
     

if do_amg
  printlnln("assemble amg_0 preconditioner for A_0 ...")
  Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);

  conds_0, conds_t = Float64[], Float64[]
  iters_0, iters_t = Int[], Int[]
  λ_lds_0, λ_lds_t = [], []
  λ_mds_0, λ_mds_t = [], []
  for ireal in 1:nreals
    printlnln("do_isotropic_elliptic_assembly for ξ_$ireal ...")
    A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                                dirichlet_inds_g2l,
                                                not_dirichlet_inds_g2l,
                                                point_markers,
                                                exp.(gs[ireal]), f, uexact)

    printlnln("assemble amg_t preconditioner for A_$ireal ...")
    Π_amg_t = @time AMGPreconditioner{SmoothedAggregation}(A);

    if save_conditioning
      apply_Π_inv_A = LinearMap(x -> Π_amg_0 \ (A * x), nothing, A.n)
      printlnln("Solve for least dominant eigenvalue of Π_amg_0^{-1} * A_$ireal ...")
      vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=SR())
      printlnln("Solve for most dominant eigenvalue of Π_amg_0^{-1} * A_$ireal ...")
      vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=LR())
      try
        λ_ld = vals_ld.eigenvalues[1].re
        λ_md = vals_md.eigenvalues[1].re
        push!(conds_0, λ_md / λ_ld)
      catch e
        push!(conds_0, -1)
      end

      apply_Π_inv_A = LinearMap(x -> Π_amg_t \ (A * x), nothing, A.n)
      printlnln("Solve for least dominant eigenvalue of Π_amg_t^{-1} * A_$ireal ...")
      vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=SR())
      printlnln("Solve for most dominant eigenvalue of Π_amg_t^{-1} * A_$ireal ...")
      vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=LR())
      try
        λ_ld = vals_ld.eigenvalues[1].re
        λ_md = vals_md.eigenvalues[1].re
        push!(conds_t, λ_md / λ_ld)
      catch e
        push!(conds_t, -1)
      end
    end

    if save_spectra
      tag = "amg_0"
      printlnln("Assemble Π_$tag^{-1} * A_$ireal ...")
      Π_inv_At = Array{Float64}(undef, A.n, A.n)
      @time for j in 1:A.n
        Π_inv_At[:, j] .= Π_amg_0 \ Array(A[:, j])
      end
      printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
      λ_ld, _ = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
      push!(λ_lds_0, λ_ld)
      npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
      printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
      λ_md, _ = @time Arpack.eigs(Π_inv_At, nev=A.n-100, which=:LM)
      push!(λ_mds_0, λ_md)
      npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)

      tag = "amg_t"
      printlnln("Assemble Π_$tag^{-1} * A_$ireal ...")
      Π_inv_At = Array{Float64}(undef, A.n, A.n)
      @time for j in 1:A.n
        Π_inv_At[:, j] .= Π_amg_t \ Array(A[:, j])
      end
      printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
      λ_ld, _ = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
      push!(λ_lds_t, λ_ld)
      npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
      printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
      λ_md, _ = @time Arpack.eigs(Π_inv_At, nev=A.n-100, which=:LM)
      push!(λ_mds_t, λ_md)
      npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)
    end

    printlnln("pcg solve of A_$ireal * u_$ireal = b_$ireal with Π_amg_0 ...")
    _, it, _ = @time pcg(A, b, zeros(A.n), Π_amg_0)
    push!(iters_0, it)

    printlnln("pcg solve of A_$ireal * u_$ireal = b_$ireal with Π_amg_t ...")
    _, it, _ = @time pcg(A, b, zeros(A.n), Π_amg_t)
    push!(iters_t, it)
  end
  if save_conditioning
    npzwrite("data/$root_fname.amg_0.conds.nreals$nreals.npz", conds_0)
    npzwrite("data/$root_fname.amg_t.conds.nreals$nreals.npz", conds_t)
  end
  npzwrite("data/$root_fname.amg_0.pcg-iters.nreals$nreals.npz", iters_0)
  npzwrite("data/$root_fname.amg_t.pcg-iters.nreals$nreals.npz", iters_t)
end



printlnln("prepare_lorasc_precond with ε = 0 ...")
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

conds_0, conds_t = Float64[], Float64[]
iters_0, iters_t = Int[], Int[]  
λ_lds_0, λ_lds_t = [], []
λ_mds_0, λ_mds_t = [], []
for ireal in 1:nreals
  printlnln("do_isotropic_elliptic_assembly for ξ_$ireal ...")
  A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(gs[ireal]), f, uexact)

  printlnln("assemble lorasc preconditioner with ε = 0.01 for A_$ireal ...")
 
  printlnln("prepare_global_schur ...")
  A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ = @time prepare_global_schur(cells,
                                                             points,
                                                             epart,
                                                             ind_Id_g2l,
                                                             ind_Γ_g2l,
                                                             node_owner,
                                                             exp.(gs[ireal]),
                                                             f,
                                                             uexact)  
                                                                                                                  
  printlnln("assemble amg preconditioners of A_IId_0 ...")
  Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IId[idom]) for idom in 1:ndom];
                                                                                                                    
  printlnln("prepare_local_schurs ...")
  A_IIdd, A_IΓdd, A_ΓΓdd, _, _ = @time prepare_local_schurs(cells,
                                                            points,
                                                            epart,
                                                            ind_Id_g2l,
                                                            ind_Γd_g2l,
                                                            ind_Γ_g2l,
                                                            node_owner,
                                                            exp.(gs[ireal]),
                                                            f,
                                                            uexact)
  
  if do_assembly_of_local_schurs
    printlnln("assemble_local_schurs ...")
    Sd_local_mat = @time assemble_local_schurs(A_IIdd, A_IΓdd, A_ΓΓdd, preconds=Π_IId)
                                                                                                                   
    printlnln("build LinearMap using assembled local schurs ...")
    S = LinearMap(x -> apply_local_schurs(Sd_local_mat,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x), nothing,
                                          n_Γ, issymmetric=true)
  else
    printlnln("build LinearMap using (p)cg solves ...")
    S = LinearMap(x -> apply_local_schurs(A_IIdd,
                                          A_IΓdd,
                                          A_ΓΓdd,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x,
                                          preconds=Π_IId), 
                                          nothing, n_Γ, issymmetric=true)
  end

  Π_lorasc_ε01_t =   @time prepare_lorasc_precond(S,
                                                  A_IId,
                                                  A_IΓd,
                                                  A_ΓΓ,
                                                  ind_Id_g2l,
                                                  ind_Γ_g2l,
                                                  not_dirichlet_inds_g2l,
                                                  ε=.01)


  if save_conditioning
    apply_Π_inv_A = LinearMap(x -> Π_lorasc_ε01_0 \ (A * x), nothing, A.n)
    printlnln("Solve for least dominant eigenvalue of Π_lorasc_ndom$ndom"*"_eps01_0^{-1} * A_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of Π_lorasc_ndom$ndom"*"_eps01_0^{-1} * A_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_0, λ_md / λ_ld)
    catch e
      push!(conds_0, -1)
    end

    apply_Π_inv_A = LinearMap(x -> Π_lorasc_ε01_t \ (A * x), nothing, A.n)
    printlnln("Solve for least dominant eigenvalue of Π_lorasc_ndom$ndom"*"_eps01_t^{-1} * A_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of Π_lorasc_ndom$ndom"*"_eps01_t^{-1} * A_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_t, λ_md / λ_ld)
    catch e
      push!(conds_t, -1)
    end
  end

  if save_spectra
    tag = "lorasc_ndom$ndom"*"_eps01_0"
    printlnln("Assemble Π_$tag^{-1} * A_$ireal ...")
    Π_inv_At = Array{Float64}(undef, A.n, A.n)
    @time for j in 1:A.n
      Π_inv_At[:, j] .= Π_lorasc_ε01_0 \ Array(A[:, j])
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
    push!(λ_lds_0, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_At, nev=A.n-100, which=:LM)
    push!(λ_mds_0, λ_md)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)

    tag = "lorasc_ndom$ndom"*"_eps01_t"
    printlnln("Assemble Π_$tag^{-1} * A_$ireal ...")
    Π_inv_At = Array{Float64}(undef, A.n, A.n)
    @time for j in 1:A.n
      Π_inv_At[:, j] .= Π_lorasc_ε01_t \ Array(A[:, j])
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
    push!(λ_lds_t, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_At, nev=A.n-100, which=:LM)
    push!(λ_mds_t, λ_md)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)
  end

  printlnln("pcg solve of A_$ireal * u_$ireal = b_$ireal with Π_lorasc_ndom$ndom"*"_eps01_0 ...")
  _, it, _ = @time pcg(A, b, zeros(A.n), Π_lorasc_ε01_0)
  push!(iters_0, it)

  printlnln("pcg solve of A_$ireal * u_$ireal = b_$ireal with Π_lorasc_ndom$ndom"*"_eps01_t ...")
  _, it, _ = @time pcg(A, b, zeros(A.n), Π_lorasc_ε01_t)
  push!(iters_t, it)
end
if save_conditioning
  npzwrite("data/$root_fname.lorasc_ndom$ndom"*"_eps01_0.conds.nreals$nreals.npz", conds_0)
  npzwrite("data/$root_fname.lorasc_ndom$ndom"*"_eps01_t.conds.nreals$nreals.npz", conds_t)
end
npzwrite("data/$root_fname.lorasc_ndom$ndom"*"_eps01_0.pcg-iters.nreals$nreals.npz", iters_0)
npzwrite("data/$root_fname.lorasc_ndom$ndom"*"_eps01_t.pcg-iters.nreals$nreals.npz", iters_t)




printlnln("prepare_lorasc_precond with ε = 0.01 ...")
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

conds_0, conds_t = Float64[], Float64[]
iters_0, iters_t = Int[], Int[]
λ_lds_0, λ_lds_t = [], []
λ_mds_0, λ_mds_t = [], []
for ireal in 1:nreals
  printlnln("do_isotropic_elliptic_assembly for ξ_$ireal ...")
  A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(gs[ireal]), f, uexact)

  printlnln("assemble lorasc preconditioner with ε = 0 for A_$ireal ...")

  printlnln("prepare_global_schur ...")
  A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ = @time prepare_global_schur(cells,
                                                             points,
                                                             epart,
                                                             ind_Id_g2l,
                                                             ind_Γ_g2l,
                                                             node_owner,
                                                             exp.(gs[ireal]),
                                                             f,
                                                             uexact)  
                                                                                                                  
  printlnln("assemble amg preconditioners of A_IId_0 ...")
  Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IId[idom]) for idom in 1:ndom];
                                                                                                                    
  printlnln("prepare_local_schurs ...")
  A_IIdd, A_IΓdd, A_ΓΓdd, _, _ = @time prepare_local_schurs(cells,
                                                            points,
                                                            epart,
                                                            ind_Id_g2l,
                                                            ind_Γd_g2l,
                                                            ind_Γ_g2l,
                                                            node_owner,
                                                            exp.(gs[ireal]),
                                                            f,
                                                            uexact)
                                                                       
  if do_assembly_of_local_schurs
    printlnln("assemble_local_schurs ...")
    Sd_local_mat = @time assemble_local_schurs(A_IIdd, A_IΓdd, A_ΓΓdd, preconds=Π_IId)
                                                                                                                    
    printlnln("build LinearMap using assembled local schurs ...")
    S = LinearMap(x -> apply_local_schurs(Sd_local_mat,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x), nothing,
                                          n_Γ, issymmetric=true)
  else
    printlnln("build LinearMap using (p)cg solves ...")
    S = LinearMap(x -> apply_local_schurs(A_IIdd,
                                          A_IΓdd,
                                          A_ΓΓdd,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x,
                                          preconds=Π_IId), 
                                          nothing, n_Γ, issymmetric=true)
  end

  Π_lorasc_ε00_t =   @time prepare_lorasc_precond(S,
                                                  A_IId,
                                                  A_IΓd,
                                                  A_ΓΓ,
                                                  ind_Id_g2l,
                                                  ind_Γ_g2l,
                                                  not_dirichlet_inds_g2l,
                                                  ε=0)
  if save_conditioning
    apply_Π_inv_A = LinearMap(x -> Π_lorasc_ε00_0 \ (A * x), nothing, A.n)
    printlnln("Solve for least dominant eigenvalue of Π_lorasc_ndom$ndom"*"_eps00_0^{-1} * A_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of Π_lorasc_ndom$ndom"*"_eps00_0^{-1} * A_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_0, λ_md / λ_ld)
    catch e
      push!(conds_0, -1)
    end

    apply_Π_inv_A = LinearMap(x -> Π_lorasc_ε00_t \ (A * x), nothing, A.n)
    printlnln("Solve for least dominant eigenvalue of Π_lorasc_ndom$ndom"*"_eps00_t^{-1} * A_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of Π_lorasc_ndom$ndom"*"_eps00_t^{-1} * A_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_t, λ_md / λ_ld)
    catch e
      push!(conds_t, -1)
    end
  end

  if save_spectra
    tag = "lorasc_ndom$ndom"*"_eps00_0"
    printlnln("Assemble Π_$tag^{-1} * A_$ireal ...")
    Π_inv_At = Array{Float64}(undef, A.n, A.n)
    @time for j in 1:A.n
      Π_inv_At[:, j] .= Π_lorasc_ε00_0 \ Array(A[:, j])
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
    push!(λ_lds_0, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_At, nev=A.n-100, which=:LM)
    push!(λ_mds_0, λ_md)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)

    tag = "lorasc_ndom$ndom"*"_eps00_t"
    printlnln("Assemble Π_$tag^{-1} * A_$ireal ...")
    Π_inv_At = Array{Float64}(undef, A.n, A.n)
    @time for j in 1:A.n
      Π_inv_At[:, j] .= Π_lorasc_ε00_t \ Array(A[:, j])
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
    push!(λ_lds_t, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_At, nev=A.n-100, which=:LM)
    push!(λ_mds_t, λ_md)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)
  end

  printlnln("pcg solve of A_$ireal * u_$ireal = b_$ireal with Π_lorasc_ndom$ndom"*"_eps00_0 ...")
  _, it, _ = @time pcg(A, b, zeros(A.n), Π_lorasc_ε00_0)
  push!(iters_0, it)

  printlnln("pcg solve of A_$ireal * u_$ireal = b_$ireal with Π_lorasc_ndom$ndom"*"_eps00_t ...")
  _, it, _ = @time pcg(A, b, zeros(A.n), Π_lorasc_ε00_t)
  push!(iters_t, it)
end
if save_conditioning
  npzwrite("data/$root_fname.lorasc_ndom$ndom"*"_eps00_0.conds.nreals$nreals.npz", conds_0)
  npzwrite("data/$root_fname.lorasc_ndom$ndom"*"_eps00_t.conds.nreals$nreals.npz", conds_t)
end
npzwrite("data/$root_fname.lorasc_ndom$ndom"*"_eps00_0.pcg-iters.nreals$nreals.npz", iters_0)
npzwrite("data/$root_fname.lorasc_ndom$ndom"*"_eps00_t.pcg-iters.nreals$nreals.npz", iters_t)




Π_bJ_0 = BJPreconditioner(ndom, A_0)

conds_0, conds_t = Float64[], Float64[]
iters_0, iters_t = Int[], Int[]
λ_lds_0, λ_lds_t = [], []
λ_mds_0, λ_mds_t = [], []
for ireal in 1:nreals
  printlnln("do_isotropic_elliptic_assembly for ξ_$ireal ...")
  A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(gs[ireal]), f, uexact)

  Π_bJ_t = BJPreconditioner(ndom, A)

  if save_conditioning
    apply_Π_inv_A = LinearMap(x -> Π_bJ_0 \ (A * x), nothing, A.n)
    printlnln("Solve for least dominant eigenvalue of Π_bJ_nb$ndom"*"_0^{-1} * A_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of Π_bJ_nb$ndom"*"_0^{-1} * A_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_0, λ_md / λ_ld)
    catch e
      push!(conds_0, -1)
    end

    apply_Π_inv_A = LinearMap(x -> Π_bJ_t \ (A * x), nothing, A.n)
    printlnln("Solve for least dominant eigenvalue of Π_bJ_nb$ndom"*"_t^{-1} * A_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of Π_bJ_nb$ndom"*"_t^{-1} * A_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_A, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_t, λ_md / λ_ld)
    catch e
      push!(conds_t, -1)
    end
  end

  if save_spectra
    tag = "bJ_nb$ndom"*"_0"
    printlnln("Assemble Π_$tag^{-1} * A_$ireal ...")
    Π_inv_At = Array{Float64}(undef, A.n, A.n)
    @time for j in 1:A.n
      Π_inv_At[:, j] .= Π_bJ_0 \ Array(A[:, j])
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
    push!(λ_lds_0, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_At, nev=A.n-100, which=:LM)
    push!(λ_mds_0, λ_md)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)

    tag = "bJ_nb$ndom"*"_t"
    printlnln("Assemble Π_$tag^{-1} * A_$ireal ...")
    Π_inv_At = Array{Float64}(undef, A.n, A.n)
    @time for j in 1:A.n
      Π_inv_At[:, j] .= Π_bJ_t \ Array(A[:, j])
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * A_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_At, nev=100, which=:SM)
    push!(λ_lds_t, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} A_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_At, nev=A.n-100, which=:LM)
    push!(λ_mds_t, λ_md)
    npzwrite("data/$root_fname.$tag" * "_As$ireal.md.eigvals.npz", λ_md)
  end

  printlnln("pcg solve of A_$ireal * u_$ireal = b_$ireal with Π_bJ_nb$ndom"*"_0 ...")
  _, it, _ = @time pcg(A, b, zeros(A.n), Π_bJ_0)
  push!(iters_0, it)

  printlnln("pcg solve of A_$ireal * u_$ireal = b_$ireal with Π_bJ_nb$ndom"*"_t ...")
  _, it, _ = @time pcg(A, b, zeros(A.n), Π_bJ_t)
  push!(iters_t, it)  
end
if save_conditioning
  npzwrite("data/$root_fname.bJ_nb$ndom"*"_0.conds.nreals$nreals.npz", conds_0)
  npzwrite("data/$root_fname.bJ_nb$ndom"*"_t.conds.nreals$nreals.npz", conds_t)
end
npzwrite("data/$root_fname.bJ_nb$ndom"*"_0.pcg-iters.nreals$nreals.npz", iters_0)
npzwrite("data/$root_fname.bJ_nb$ndom"*"_t.pcg-iters.nreals$nreals.npz", iters_t)



"""

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

  
  #printlnln("ld-def-amg_0-pcg solve of A * u = b ...")
  #u, it, _ = @time defpcg(As[ireal], bs[ireal], zeros(As[ireal].n), Φ_ld, Π_amg_0);
  #space_println("n = $(As[ireal].n), nev = $nev (ld), iter = $it")

  #printlnln("md-def-amg_0-pcg solve of A * u = b ...")
  #u, it, _ = @time defpcg(As[ireal], bs[ireal], zeros(As[ireal].n), Φ_md, Π_amg_0);
  #space_println("n = $(As[ireal].n), nev = $nev (md), iter = $it")
  

  printlnln("lorasc_0-pcg solve of A * u = b ...")
  u, it, _ = @time pcg(As[ireal], bs[ireal], zeros(As[ireal].n), Π_lorasc_0)
  space_println("n = $(As[ireal].n), iter = $it")

  
  #printlnln("ld-def-lorasc_0-pcg solve of A * u = b ...")
  #u, it, _ = @time defpcg(As[ireal], bs[ireal], zeros(As[ireal].n), Φ_ld, Π_lorasc_0);
  #space_println("n = $(As[ireal].n), nev = $nev (ld), iter = $it")
end
"""