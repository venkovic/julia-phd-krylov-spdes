using TriangleMesh
push!(LOAD_PATH, "./Fem/")
using Fem

poly = polygon_Lshape()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)
fig = plot_TriMesh(mesh)

nnode = mesh.n_point
p = mesh.point

nel = mesh.n_cell
nodes = mesh.cell

function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  L = .1
  return exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end
  
C = @time do_mass_covariance_assembly(mesh.cell, mesh.point, cov)
M = @time get_mass_matrix(mesh.cell, mesh.point)