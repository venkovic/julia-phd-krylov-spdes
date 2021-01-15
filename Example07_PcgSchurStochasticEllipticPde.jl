push!(LOAD_PATH, "./Fem/")
import Pkg
Pkg.activate(".")
using Fem
using NPZ
using IterativeSolvers
using Preconditioners
using LinearMaps

model = "SExp"
sig2 = 1.
L = .1

tentative_nnode = 400_000
load_existing_mesh = true

ndom = 400
load_existing_partition = true

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

ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, node_owner, 
elemd, node_Γ, node_Id, nnode_Id = set_subdomains(cells, cell_neighbors, epart,
                                                  npart, dirichlet_inds_g2l)

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

# The eigenfunctions obtained by domain 
# decomposition are not perfectly orthogonal
χ = get_kl_coordinates(g, Λ, Ψ, M)  
println("extrema(ξ - χ) = $(extrema(ξ - χ))")

print("in-place draw ...")
@time draw!(Λ, Ψ, ξ, g)

print("do_schur_assembly ...")
A_IId, A_IΓd, A_ΓΓd, A_ΓΓ, b_Id, b_Γ = @time do_schur_assembly(cells,
                                                               points,
                                                               epart,
                                                               ind_Id_g2l,
                                                               ind_Γd_g2l,
                                                               ind_Γ_g2l,
                                                               node_owner,
                                                               exp.(g),
                                                               f,
                                                               uexact)


print("assemble amg preconditioners ...")
Π_IId = @time [AMGPreconditioner{SmoothedAggregation}(A_IId[idom])
               for idom in 1:ndom];

n_Γ, _ = size(A_ΓΓ)
S = LinearMap(x -> apply_schur(A_IId, A_IΓd, A_ΓΓ, x), n_Γ, issymmetric=true)

n_Γd = [size(A_ΓΓd[idom])[1] for idom in 1:ndom]
Sd = [LinearMap(x -> apply_local_schur(A_IId[idom], A_IΓd[idom], A_ΓΓd[idom], x), 
                     n_Γ, issymmetric=true) for idom in 1:ndom]

#S = LinearMap(x -> apply_schur(A_IId, A_IΓd, A_ΓΓ, x, Π_IId), n_Γ, issymmetric=true)

print("get_schur_rhs ...")
b_schur = @time get_schur_rhs(b_Id, A_IId, A_IΓd, b_Γ)

print("solve for u_Γ ...")
u_Γ = @time IterativeSolvers.cg(S, b_schur)

print("get_subdomain_solutions ...")
u_Id = @time get_subdomain_solutions(u_Γ, A_IId, A_IΓd, b_Id)

u_with_dd = merge_subdomain_solutions(u_Γ, u_Id, node_Γ, node_Id,
                                      dirichlet_inds_l2g, uexact,
                                      points)