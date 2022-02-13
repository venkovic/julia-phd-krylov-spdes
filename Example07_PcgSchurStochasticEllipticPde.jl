push!(LOAD_PATH, "./Utils/")
push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
import Pkg
Pkg.activate(".")
using Fem
using RecyclingKrylovSolvers
using ArnoldiMethod

using Utils: space_println, printlnln

using LinearMaps
using Preconditioners
using NPZ
import Arpack

model = "SExp"
sig2 = 1.
L = .1

nreals = 1_0#00

tentative_nnode = 4_000 # 4_000, 8_000, 16_000, 32_000, 64_000, 128_000
load_existing_mesh = true
save_spectra = true
save_conditioning = false
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


#
# Operator assembly for ξ = 0
#
g_0 = copy(g); g_0 .= 0.

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

printlnln("prepare_local_schurs ...")
A_IIdd_0, A_IΓdd_0, A_ΓΓdd_0, b_Idd_0, b_Γ_0 = @time prepare_local_schurs(cells,
                                                                          points,
                                                                          epart,
                                                                          ind_Id_g2l,
                                                                          ind_Γd_g2l,
                                                                          ind_Γ_g2l,
                                                                          node_owner,
                                                                          exp.(g_0),
                                                                          f,
                                                                          uexact)

printlnln("assemble amg preconditioners of A_IId ...")
Π_IId_0 = @time [AMGPreconditioner{SmoothedAggregation}(A_IIdd_0[idom])
               for idom in 1:ndom];

if do_assembly_of_local_schurs
  printlnln("assemble_local_schurs ...")
  Sd_local_mat_0 = @time assemble_local_schurs(A_IIdd_0, A_IΓdd_0, A_ΓΓdd_0, preconds=Π_IId_0)
                                                                              
  S_0 = LinearMap(x -> apply_local_schurs(Sd_local_mat_0,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x),
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
# Realization assemblies for random ξ_t's
#
printlnln("in-place draw of ξ ...")
gs = [copy(g) for _ in 1:nreals]
for ireal in 1:nreals
  @time draw!(Λ, Ψ, ξ, g)
  gs[ireal] .= g
end



printlnln("prepare_neumann_neumann_schur_precond using S_local_mat_0 ...")
ΠSnn_0 = @time prepare_neumann_neumann_schur_precond(Sd_local_mat_0,
                                                     ind_Γd_Γ2l,
                                                     node_Γ_cnt)

conds_0, conds_t = Float64[], Float64[]
iters_0, iters_t = Int[], Int[]
λ_lds_0, λ_lds_t = [], []
λ_mds_0, λ_mds_t = [], []
for ireal in 1:nreals  
  printlnln("prepare_local_schurs ...")
  A_IIdd, A_IΓdd, A_ΓΓdd, b_Idd, b_Γ = @time prepare_local_schurs(cells,
                                                                  points,
                                                                  epart,
                                                                  ind_Id_g2l,
                                                                  ind_Γd_g2l,
                                                                  ind_Γ_g2l,
                                                                  node_owner,
                                                                  exp.(gs[ireal]),
                                                                  f,
                                                                  uexact)
  
  printlnln("assemble amg preconditioners of A_IId ...")
  Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IIdd[idom])
                 for idom in 1:ndom];
  
  printlnln("get_schur_rhs ...")
  b_schur = @time get_schur_rhs(b_Idd, A_IIdd, A_IΓdd, b_Γ, ind_Γd_Γ2l, preconds=Π_IId)
  
  if do_assembly_of_local_schurs
    printlnln("assemble_local_schurs ...")
    Sd_local_mat = @time assemble_local_schurs(A_IIdd, A_IΓdd, A_ΓΓdd, preconds=Π_IId)
                                                                                
    S = LinearMap(x -> apply_local_schurs(Sd_local_mat,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x),
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
  
  printlnln("prepare_neumann_neumann_schur_precond using S_local_mat ...")
  ΠSnn_t = @time prepare_neumann_neumann_schur_precond(Sd_local_mat,
                                                       ind_Γd_Γ2l,
                                                       node_Γ_cnt)


  if save_conditioning
    apply_Π_inv_S = LinearMap(x -> ΠSnn_0 \ (S * x), nothing, S.N)
    println(apply_Π_inv_S * zeros(S.N))
    printlnln("Solve for least dominant eigenvalue of ΠSnn_0^{-1} * S_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_S, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of ΠSnn_0^{-1} * S_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_S, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_0, λ_md / λ_ld)
    catch e
      push!(conds_0, -1)
    end

    apply_Π_inv_S = LinearMap(x -> ΠSnn_t \ (S * x), nothing, S.N)
    printlnln("Solve for least dominant eigenvalue of ΠSnn_t^{-1} S_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_S, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of ΠSnn_t^{-1} * S_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_S, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_t, λ_md / λ_ld)
    catch e
      push!(conds_t, -1)
    end
  end

  if save_spectra
    tag = "neumann-neumann_0"
    printlnln("Assemble Π_$tag^{-1} * S_$ireal ...")
    Π_inv_St = Array{Float64}(undef, S.N, S.N)
    @time for j in 1:S.N
      ej = zeros(S.N)
      ej[j] = 1
      Π_inv_St[:, j] .= ΠSnn_0 \ (S * ej)
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * S_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_St, nev=100, which=:SM)
    push!(λ_lds_0, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_Ss$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} S_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_St, nev=S.N-100, which=:LM)
    push!(λ_mds_0, λ_md)
    npzwrite("data/$root_fname.$tag" * "_Ss$ireal.md.eigvals.npz", λ_md)

    tag = "neumann-neumann_t"
    printlnln("Assemble Π_$tag^{-1} * S_$ireal ...")
    Π_inv_St = Array{Float64}(undef, S.N, S.N)
    @time for j in 1:S.N
      ej = zeros(S.N)
      ej[j] = 1
      Π_inv_St[:, j] .= ΠSnn_t \ (S * ej)
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * S_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_St, nev=100, which=:SM)
    push!(λ_lds_t, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_Ss$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} S_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_St, nev=S.N-100, which=:LM)
    push!(λ_mds_t, λ_md)
    npzwrite("data/$root_fname.$tag" * "_Ss$ireal.md.eigvals.npz", λ_md)
  end

  printlnln("neumann-neumann_0-pcg solve of u_Γ s.t. S_$ireal * u_Γ = b_schur ...")
  u_Γ, it, _ = @time pcg(S, b_schur, zeros(S.N), ΠSnn_0);
  push!(iters_0, it)

  printlnln("neumann-neumann_t-pcg solve of u_Γ s.t. S_$ireal * u_Γ = b_schur ...")
  u_Γ, it, _ = @time pcg(S, b_schur, zeros(S.N), ΠSnn_t);
  push!(iters_t, it)
end
if save_conditioning
  npzwrite("data/$root_fname.neumann-neumann_0.conds.nreals$nreals.npz", conds_0)
  npzwrite("data/$root_fname.neumann-neumann_t.conds.nreals$nreals.npz", conds_t)
end
npzwrite("data/$root_fname.neumann-neumann_0.pcg-iters.nreals$nreals.npz", iters_0)
npzwrite("data/$root_fname.neumann-neumann_t.pcg-iters.nreals$nreals.npz", iters_t)




conds_0, conds_t = Float64[], Float64[]
iters_0, iters_t = Int[], Int[]
λ_lds_0, λ_lds_t = [], []
λ_mds_0, λ_mds_t = [], []
for ireal in 1:nreals  
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

  printlnln("prepare_local_schurs ...")
  A_IIdd, A_IΓdd, A_ΓΓdd, b_Idd, b_Γ = @time prepare_local_schurs(cells,
                                                                  points,
                                                                  epart,
                                                                  ind_Id_g2l,
                                                                  ind_Γd_g2l,
                                                                  ind_Γ_g2l,
                                                                  node_owner,
                                                                  exp.(gs[ireal]),
                                                                  f,
                                                                  uexact)
  
  printlnln("assemble amg preconditioners of A_IId ...")
  Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IIdd[idom])
                 for idom in 1:ndom];
  
  printlnln("get_schur_rhs ...")
  b_schur = @time get_schur_rhs(b_Idd, A_IIdd, A_IΓdd, b_Γ, ind_Γd_Γ2l, preconds=Π_IId)
  
  if do_assembly_of_local_schurs
    printlnln("assemble_local_schurs ...")
    Sd_local_mat = @time assemble_local_schurs(A_IIdd, A_IΓdd, A_ΓΓdd, preconds=Π_IId)
                                                                                
    S = LinearMap(x -> apply_local_schurs(Sd_local_mat,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x),
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

  if save_conditioning
    apply_Π_inv_S = LinearMap(x -> ΠSnn_local_mat_0 \ (S * x), nothing, S.N)
    println(apply_Π_inv_S * zeros(S.N))
    printlnln("Solve for least dominant eigenvalue of ΠSnn_0^{-1} * S_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_S, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of ΠSnn_0^{-1} * S_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_S, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_0, λ_md / λ_ld)
    catch e
      push!(conds_0, -1)
    end

    apply_Π_inv_S = LinearMap(x -> ΠSnn_local_mat_t \ (S * x), nothing, S.N)
    printlnln("Solve for least dominant eigenvalue of ΠSnn_t^{-1} S_$ireal ...")
    vals_ld, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_S, nev=1, tol=1e-6, which=SR())
    printlnln("Solve for most dominant eigenvalue of ΠSnn_t^{-1} * S_$ireal ...")
    vals_md, _ = @time ArnoldiMethod.partialschur(apply_Π_inv_S, nev=1, tol=1e-6, which=LR())
    try
      λ_ld = vals_ld.eigenvalues[1].re
      λ_md = vals_md.eigenvalues[1].re
      push!(conds_t, λ_md / λ_ld)
    catch e
      push!(conds_t, -1)
    end
  end

  if save_spectra
    tag = "A_GG_0"
    printlnln("Assemble Π_$tag^{-1} * S_$ireal ...")
    Π_inv_St = Array{Float64}(undef, S.N, S.N)
    @time for j in 1:S.N
      ej = zeros(S.N)
      ej[j] = 1
      Π_inv_St[:, j] .= A_ΓΓ_0  \ (S * ej)
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * S_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_St, nev=100, which=:SM)
    push!(λ_lds_0, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_Ss$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} S_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_St, nev=S.N-100, which=:LM)
    push!(λ_mds_0, λ_md)
    npzwrite("data/$root_fname.$tag" * "_Ss$ireal.md.eigvals.npz", λ_md)

    tag = "A_GG_t"
    printlnln("Assemble Π_$tag^{-1} * S_$ireal ...")
    Π_inv_St = Array{Float64}(undef, S.N, S.N)
    @time for j in 1:S.N
      ej = zeros(S.N)
      ej[j] = 1
      Π_inv_St[:, j] .= A_ΓΓ \ (S * ej)
    end
    printlnln("solve for least dominant eigvecs of Π_$tag^{-1} * S_$ireal ...")
    λ_ld, _ = @time Arpack.eigs(Π_inv_St, nev=100, which=:SM)
    push!(λ_lds_t, λ_ld)
    npzwrite("data/$root_fname.$tag" * "_Ss$ireal.ld.eigvals.npz", λ_ld)
    printlnln("solve for most dominant eigvecs of Π_$tag^{-1} S_$ireal ...")
    λ_md, _ = @time Arpack.eigs(Π_inv_St, nev=S.N-100, which=:LM)
    push!(λ_mds_t, λ_md)
    npzwrite("data/$root_fname.$tag" * "_Ss$ireal.md.eigvals.npz", λ_md)
  end

  printlnln("A_GG_0-pcg solve of u_Γ s.t. S_$ireal * u_Γ = b_schur ...")
  u_Γ, it, _ = @time pcg(S, b_schur, zeros(S.N), A_ΓΓ_0);
  push!(iters_0, it)

  printlnln("A_GG_t-pcg solve of u_Γ s.t. S_$ireal * u_Γ = b_schur ...")
  u_Γ, it, _ = @time pcg(S, b_schur, zeros(S.N), A_ΓΓ);
  push!(iters_t, it)
end
if save_conditioning
  npzwrite("data/$root_fname.A_GG_0.conds.nreals$nreals.npz", conds_0)
  npzwrite("data/$root_fname.A_GG_t.conds.nreals$nreals.npz", conds_t)
end
npzwrite("data/$root_fname.A_GG_0.pcg-iters.nreals$nreals.npz", iters_0)
npzwrite("data/$root_fname.A_GG_t.pcg-iters.nreals$nreals.npz", iters_t)































""" 
printlnln("do_isotropic_elliptic_assembly ...")
A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            exp.(g), f, uexact)

printlnln("assemble amg preconditioner of A ...")
Π = @time AMGPreconditioner{SmoothedAggregation}(A)
                                                            
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