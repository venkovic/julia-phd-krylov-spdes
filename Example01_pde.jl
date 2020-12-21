using TriangleMesh
push!(LOAD_PATH, "./Fem/")
using Fem

poly = polygon_Lshape()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)
#fig = plot_TriMesh(mesh)

nnode = mesh.n_point
nbdnseg = mesh.n_segment
nel = mesh.n_cell

function a(x::Float64, y::Float64)
  return .1 + .0001 * x * y
end
  
function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx, yy)
  return .734
end

A, b = @time do_isotropic_elliptic_assembly(mesh.cell, mesh.point, a, f)
println(A.nzval)
A, b = apply_dirichlet(mesh.segment, mesh.point, A, b, uexact)
println(A.nzval)

using IterativeSolvers
u = IterativeSolvers.cg(A, b)

