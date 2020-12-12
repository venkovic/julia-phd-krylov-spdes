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

A, b = @time do_assembly(mesh.cell, mesh.point)
#A, b = apply_dirichlet(e, p, nnode, A, b, g1)
#u = solve(gk, gf)

