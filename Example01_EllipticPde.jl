using TriangleMesh
using NPZ
push!(LOAD_PATH, "./Fem/")
using Fem

tentative_nnode = 100_000
mesh = get_mesh(tentative_nnode)
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

function uexact(xx::Float64, yy::Float64)
  return 3.
end

dirichlet_inds_g2l, not_dirichlet_inds_g2l,
dirichlet_inds_l2g, not_dirichlet_inds_l2g = 
get_dirichlet_inds(mesh.point, mesh.point_marker)

A, b = @time do_isotropic_elliptic_assembly(mesh.cell, mesh.point,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            mesh.point_marker,
                                            a, f, uexact)

npzwrite("cells.npz", mesh.cell' .- 1)
npzwrite("points.npz", mesh.point')
npzwrite("point_markers.npz", mesh.point_marker')

using IterativeSolvers
u_no_dirichlet = @time IterativeSolvers.cg(A, b)

u = @time append_bc(dirichlet_inds_l2g, not_dirichlet_inds_l2g,
                    u_no_dirichlet, mesh.point, uexact)
npzwrite("u.npz", u)