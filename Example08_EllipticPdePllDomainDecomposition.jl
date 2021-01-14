using Distributed

addprocs(([("marcel", 4)]), tunnel=true)
addprocs(([("andrew", 3)]), tunnel=true)
#addprocs(([("moorcock", 4)]), tunnel=true)
addprocs(3)

@everywhere begin
  push!(LOAD_PATH, "./Fem/")
  import Pkg
  Pkg.activate(".")
end

@everywhere begin 
  using Fem
  using Distributed
  using Distributions
  using DistributedOperations
end








import LinearAlgebra
using NPZ







using LinearMaps
using IterativeSolvers
using Preconditioners








@everywhere begin
  ndom = 400
  tentative_nnode = 100_000
end

mesh = get_mesh(tentative_nnode)

dirichlet_inds_g2l, not_dirichlet_inds_g2l,
dirichlet_inds_l2g, not_dirichlet_inds_l2g = 
get_dirichlet_inds(mesh.point, mesh.point_marker)

epart, npart = mesh_partition(mesh, ndom)
ind_Id_g2l, ind_Γ_g2l, node_owner, 
elemd, node_Γ, node_Id, nnode_Id = set_subdomains(mesh, epart, npart, dirichlet_inds_g2l)

bcast(mesh, procs())
bcast(epart, procs())
bcast(ind_Id_g2l, procs())
bcast(ind_Γ_g2l, procs())
bcast(node_owner, procs())

@everywhere function a(x::Float64, y::Float64)
  return .1 + .0001 * x * y
end
  
@everywhere function f(x::Float64, y::Float64)
  return -1.
end

@everywhere function uexact(xx::Float64, yy::Float64)
  return .734
end

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
S = LinearMap(x -> apply_schur(A_IId, A_IΓd, A_ΓΓ, x), n_Γ, issymmetric=true)
#S = LinearMap(x -> apply_schur(A_IId, A_IΓd, A_ΓΓ, x, Π_IId), n_Γ, issymmetric=true)
print("get_schur_rhs ...")
b_schur = @time get_schur_rhs(b_Id, A_IId, A_IΓd, b_Γ)

print("solve for u_Γ ...")
u_Γ = @time IterativeSolvers.cg(S, b_schur)
print("get_subdomain_solutions ...")
u_Id = @time get_subdomain_solutions(u_Γ, A_IId, A_IΓd, b_Id)

u_with_dd = merge_subdomain_solutions(u_Γ, u_Id, node_Γ, node_Id,
                                      dirichlet_inds_l2g, uexact,
                                      mesh.point)















































