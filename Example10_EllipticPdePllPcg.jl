push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: pcg
using Utils: space_println, printlnln

#using IterativeSolvers: cg
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using NPZ: npzwrite

tentative_nnode = 2_000_000
load_existing_mesh = false

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

space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

printlnln("assemble linear system ...")
A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                            dirichlet_inds_g2l,
                                            not_dirichlet_inds_g2l,
                                            point_markers,
                                            a, f, uexact)

printlnln("assemble amg preconditioner ...")
Π = @time AMGPreconditioner{SmoothedAggregation}(A);

printlnln("amg-pcg solve ...")
#u_no_dirichlet, it, _  = @time cg(A, b, Pl=Π)
u_no_dirichlet, it, _  = @time pcg(A, b, M=Π)
space_println("n = $(A.n), iter = $it")

printlnln("append dirchlet nodes to solution ...")
u = @time append_bc(dirichlet_inds_l2g, not_dirichlet_inds_l2g,
                    u_no_dirichlet, points, uexact)
npzwrite("data/DoF$tentative_nnode.u.npz", u)