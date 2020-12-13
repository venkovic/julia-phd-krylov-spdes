using TriangleMesh
push!(LOAD_PATH, ".");
using Fem

poly = polygon_Lshape()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)
#fig = plot_TriMesh(mesh)

nnode = mesh.n_point
p = mesh.point

nel = mesh.n_cell
nodes = mesh.cell

function a(x::Float64, y::Float64)
    return 1. + x * y
end
  
function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx, yy)
    return .734
end

A, b = @time do_assembly(mesh.cell, mesh.point, a, f)
#A, b = apply_dirichlet(e, mesh.point, A, b, uexact)
#u = solve(A, b)

