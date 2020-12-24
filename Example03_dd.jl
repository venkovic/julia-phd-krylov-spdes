using TriangleMesh
using NPZ
push!(LOAD_PATH, "./Fem/")
using Fem

poly = polygon_unitSquare()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

ndom = 400
epart, npart = mesh_partition(mesh, ndom)
elemd, node_Γ, node_Id, ind_Id, nn_Id, is_on_Γ = set_subdomains(mesh, epart, npart)

function a(x::Float64, y::Float64)
  return .1 + .0001 * x * y
end
  
function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx, yy)
  return .734
end

#A_IId = do_IId_assembly(elemd, cell, point, is_on_Γ, ind_Id, nn_Id, a, f, 1)
#do_IΓ_assembly()
#do_ΓΓ_assembly()

@time npzwrite("cells.npz", mesh.cell' .- 1)
@time npzwrite("points.npz", mesh.point')

@time npzwrite("epart.npz", epart .- 1)
@time npzwrite("nodes_at_interface.npz", node_Γ .- 1)
for id in 1:ndom
  npzwrite("nodes_inside_$id.npz", node_Id[id] .- 1)
end