push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./Utils/")
push!(LOAD_PATH, "./MyPreconditioners/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: cg, pcg, defpcg

using Utils: space_println, printlnln

using MyPreconditioners: BJPreconditioner
using Preconditioners: AMGPreconditioner, SmoothedAggregation
using NPZ: npzread, npzwrite
using Random: seed!; seed!(123_456);
using LinearMaps: LinearMap, cholesky
using SparseArrays: SparseMatrixCSC
import Arpack

tentative_nnode = 100_000
load_existing_mesh = false

nreals = 1000

model = "SExp"
sig2 = 1.
L = .1 # .1, .01, .001
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

if load_existing_mesh
  cells, points, point_markers, cell_neighbors = load_mesh(tentative_nnode)
else
  mesh = get_mesh(tentative_nnode)
  cells = mesh.cell
  points = mesh.point
  point_markers = mesh.point_marker
  cell_neighbors = mesh.cell_neighbor
end

dirichlet_inds_g2l, not_dirichlet_inds_g2l,
dirichlet_inds_l2g, not_dirichlet_inds_l2g =
get_dirichlet_inds(points, point_markers)

function f(x::Float64, y::Float64)
  return -1.
end
  
function uexact(xx::Float64, yy::Float64)
  return .734
end

println()
space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")

M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
nKL = length(Λ)
#
# Realization assemblies for random ξ_t's
#

printlnln("in-place draw of ξ ...")
ξ, g = draw(Λ, Ψ)

iters_amg = Array{Int,2}(undef, nreals, nKL)
iters_chol = Array{Int,2}(undef, nreals, nKL)

for ireal in 1:nreals
  printlnln("do_isotropic_elliptic_assembly of fully resolved coefficient field ...")
  A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(g), f, uexact)


  ξ_trunc = copy(ξ)

  for k in 1:nKL

    ξ_trunc[1:k] = ξ[1:k]
    ξ_trunc[k+1:end] .= 0 
  
    set!(Λ, Ψ, ξ_trunc, g)
  
    printlnln("do_isotropic_elliptic_assembly of truncated coefficient field ...")
    A_trunc, _ = @time do_isotropic_elliptic_assembly(cells, points,
                                                    dirichlet_inds_g2l,                                                
                                                    not_dirichlet_inds_g2l,
                                                    point_markers,
                                                    exp.(g), f, uexact)


    printlnln("assemble cholesky factorization of A_trunc ...")
    chol = @time cholesky(A_trunc);

    printlnln("assemble amg_t preconditioner for A_trunc ...")
    Π_amg = @time AMGPreconditioner{SmoothedAggregation}(A_trunc);
                                              
    printlnln("pcg solve of A * u = b with Π_amg_trunc ...")
    _, iters_amg[ireal, k], _ = @time pcg(A, b, zeros(A.n), Π_amg)
    println(" k = $k / $nKL, it = $(iters_amg[ireal, k])")

    printlnln("pcg solve of A * u = b with chol_trunc ...")
    _, iters_chol[ireal, k], _ = @time pcg(A, b, zeros(A.n), chol)
    println("it = $(iters_chol[ireal, k])")
  end

  @time draw!(Λ, Ψ, ξ, g)
end

npzwrite("data/Example19.amg.kl.iters.npz", iters_amg)
npzwrite("data/Example19.chol.kl.iters.npz", iters_chol)




