using TriangleMesh
using NPZ
push!(LOAD_PATH, "./Fem/")
using Fem

poly = polygon_unitSquare()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

nd = 400
epart, npart = mesh_partition(mesh, nd)
nodes_at_interface = set_subdomains(mesh, epart)

function a(x::Float64, y::Float64)
  return .1 + .0001 * x * y
end
  
function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx, yy)
  return .734
end

@time npzwrite("cells.npz", mesh.cell' .- 1)
@time npzwrite("points.npz", mesh.point')

@time npzwrite("epart.npz", epart .- 1)
@time npzwrite("nodes_at_interface.npz", nodes_at_interface .- 1)