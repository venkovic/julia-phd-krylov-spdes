push!(LOAD_PATH, "./Fem/")
import Pkg
Pkg.activate(".")
using Fem
using NPZ

tentative_nnode = 1_000_000
load_existing_mesh = true

if load_existing_mesh
  cells, points, point_markers, cell_neighbors = load_mesh(tentative_nnode)
else
  mesh = get_mesh(tentative_nnode)
  save_mesh(mesh, tentative_nnode)
  cells = mesh.cell
  points = mesh.point
  point_markers = mesh.point_marker
  cell_neighbors = mesh.cell_neighbor
end

dirichlet_inds_g2l, not_dirichlet_inds_g2l,
dirichlet_inds_l2g, not_dirichlet_inds_l2g = 
get_dirichlet_inds(points, point_markers)

function a(x::Float64, y::Float64)
  return .1 + .0001 * x * y
end
  
function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx::Float64, yy::Float64)
  return 3.
end

println("nnode = $(size(points)[2])")
println("nel = $(size(cells)[2])")

print("assemble linear system ...")
A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            a, f, uexact)


using IterativeSolvers
using Preconditioners

print("assemble amg preconditioner ...")
Π = @time AMGPreconditioner{SmoothedAggregation}(A);

print("cg solve ...")
u_no_dirichlet = @time IterativeSolvers.cg(A, b, Pl=Π)

print("append dirchlet nodes to solution ...")
u = @time append_bc(dirichlet_inds_l2g, not_dirichlet_inds_l2g,
                    u_no_dirichlet, points, uexact)
npzwrite("data/DoF$tentative_nnode.u.npz", u)