using TriangleMesh
using NPZ
push!(LOAD_PATH, "./Fem/")
using Fem


poly = polygon_unitSquare()
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

# APPLYING BC IS WAY TOO SLOW AS IT IS!!!
A, b = @time apply_dirichlet(mesh.segment, mesh.point, A, b, uexact)

@time npzwrite("cells.npz", mesh.cell' .- 1)
@time npzwrite("points.npz", mesh.point')

using IterativeSolvers
u = @time IterativeSolvers.cg(A, b)
@time npzwrite("u.npz", u)