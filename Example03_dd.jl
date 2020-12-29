using TriangleMesh
using NPZ
push!(LOAD_PATH, "./Fem/")
using Fem
using LinearMaps
using IterativeSolvers

poly = polygon_unitSquare()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

ndom = 400
epart, npart = mesh_partition(mesh, ndom)
ind_Id, ind_Γ, is_on_Γ, elemd, node_Γ, node_Id, nn_Id = set_subdomains(mesh, epart, npart)
# only ind_Id, ind_Γ, is_on_Γ are essential

function a(x::Float64, y::Float64)
  return .1 + .0001 * x * y
end
  
function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx, yy)
  return .734
end

A_IId, A_IΓd, A_ΓΓ, b_I, b_Γ = do_schur_assembly(mesh.cell, mesh.point, epart, ind_Id, ind_Γ, is_on_Γ, a, f)
n_Γ, _ = size(A_ΓΓ)
S = LinearMap(x -> apply_schur(A_IId, A_IΓd, A_ΓΓ, x), n_Γ, issymmetric=true)


# To do: - apply BC
#        - get ̂b, i.e., RHS and Schur RHS
#
u_Γ = rand(n_Γ)
cg!(S, u_Γ, ̂b)
# Compute u_Id
# Compare with regular solve

#using Distributed
#addprocs(2)
#addprocs(([("marcel", 4)]), tunnel=true)
#addprocs(([("andrew", 4)]), tunnel=true)

@time npzwrite("cells.npz", mesh.cell' .- 1)
@time npzwrite("points.npz", mesh.point')

#@time npzwrite("epart.npz", epart .- 1)
#@time npzwrite("nodes_at_interface.npz", node_Γ .- 1)
#for id in 1:ndom
#  npzwrite("nodes_inside_$id.npz", node_Id[id] .- 1)
#end

