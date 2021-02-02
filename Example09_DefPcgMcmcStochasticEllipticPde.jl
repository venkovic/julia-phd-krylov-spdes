push!(LOAD_PATH, "./Fem/")
push!(LOAD_PATH, "./RecyclingKrylovSolvers/")
push!(LOAD_PATH, "./Utils/")
import Pkg
Pkg.activate(".")

using Fem
using RecyclingKrylovSolvers: cg, pcg, defpcg, eigpcg, eigdefpcg
using Utils: space_println, printlnln

using Preconditioners: AMGPreconditioner, SmoothedAggregation
using NPZ: npzread
using Random: seed!; seed!(481_456);
using LinearMaps: LinearMap

tentative_nnode = 50_000
load_existing_mesh = false

ndom = 40
load_existing_partition = false

nvec = ndom + 5
spdim = floor(Int, 2.5 * nvec)

do_lorasc_0_pcg = false

verbose = true

model = "SExp"
sig2 = 1.
L = .1

nsmp = 30

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

n = not_dirichlet_inds_g2l.count
nnode = mesh.n_point

function f(x::Float64, y::Float64)
  return -1.
end
  
function uexact(xx::Float64, yy::Float64)
  return .734
end

println()
space_println("nnode = $(size(points)[2])")
space_println("nel = $(size(cells)[2])")


#
# Assembly of constant preconditioners
#
g_0 = zeros(Float64, nnode)

printlnln("do_isotropic_elliptic_assembly for ξ = 0 ...")
A_0, _ = @time do_isotropic_elliptic_assembly(cells, points,
                                              dirichlet_inds_g2l,
                                              not_dirichlet_inds_g2l,
                                              point_markers,
                                              exp.(g_0), f, uexact)

printlnln("prepare amg_0 preconditioner for A_0 ...")
Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);

printlnln("prepare_lorasc_precond for A_0 ...")
Π_lorasc_0 = @time prepare_lorasc_precond(tentative_nnode,
                                          ndom,
                                          cells,
                                          points,
                                          cell_neighbors,
                                          exp.(g_0),
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          f,
                                          uexact)


#
# Load kl representation and prepare mcmc sampler
# 
M = get_mass_matrix(cells, points)
Λ = npzread("data/$root_fname.kl-eigvals.npz")
Ψ = npzread("data/$root_fname.kl-eigvecs.npz")
                                              
sampler = prepare_mcmc_sampler(Λ, Ψ)

function test01(Π_amg_0,
                Π_lorasc_0::LorascPreconditioner,
                sampler::McmcSampler,
                verbose::Bool,
                do_lorasc_0_pcg::Bool,
                nsmp::Int)

  verbose ? println("\n1 / $nsmp") : nothing
  verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
  Δt = @elapsed A, b = do_isotropic_elliptic_assembly(cells, points,
                                                      dirichlet_inds_g2l,
                                                      not_dirichlet_inds_g2l,
                                                      point_markers,
                                                      exp.(sampler.g),
                                                      f, uexact)
  verbose ? println("$Δt seconds") : nothing

  x = zeros(Float64, A.n)

  verbose ? print("amg_0-pcg of A * u = b ... ") : nothing
  Δt = @elapsed _, it, _  = pcg(A, b, M=Π_amg_0)
  verbose ? println("$Δt seconds, iter = $it") : nothing

  if do_lorasc_0_pcg
    print("lorasc_0-pcg solve of A * u = b ...")
    Δt = @elapsed _, it, _ = pcg(A, b, M=Π_lorasc_0)
    verbose ? println("$Δt seconds, iter = $it") : nothing
  end

  print("lorasc_0-eigpcg solve of A * u = b ...")
  Δt = @elapsed _, it, _, W = eigpcg(A, b, x, Π_lorasc_0, nvec, spdim)
  verbose ? println("$Δt seconds, iter = $it") : nothing

  #
  # Sample ξ by mcmc and solve linear systems by def-pcg with 
  # online eigenvectors approximation
  #
  cnt_reals = 1
  for s in 2:nsmp

    verbose ? print("\n$s / $nsmp") : nothing
    cnt_reals += draw!(sampler)
    verbose ? println(" (acceptance frequency: $(s / cnt_reals))") : nothing

    verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
    Δt = @elapsed update_isotropic_elliptic_assembly!(A, b,
                                                      cells, points,
                                                      dirichlet_inds_g2l,
                                                      not_dirichlet_inds_g2l,
                                                      point_markers,
                                                      exp.(sampler.g),
                                                      f, uexact)
    verbose ? println("$Δt seconds") : nothing

    verbose ? print("amg_0-pcg of A * u = b ... ") : nothing
    Δt = @elapsed _, it, _ = pcg(A, b, M=Π_amg_0)
    verbose ? println("$Δt seconds, iter = $it") : nothing

    if do_lorasc_0_pcg
      print("lorasc_0-pcg solve of A * u = b ...")
      Δt = @elapsed _, it, _ = pcg(A, b, M=Π_lorasc_0)
      verbose ? println("$Δt seconds, iter = $it") : nothing
    end

    print("lorasc_0-eigdefpcg solve of A * u = b ...")
    Δt = @elapsed _, it, _, W = eigdefpcg(A, b, x, Π_lorasc_0, W, spdim)
    verbose ? println("$Δt seconds, iter = $it") : nothing

  end
end

test01(Π_amg_0,
       Π_lorasc_0,
       sampler,
       verbose,
       do_lorasc_0_pcg,
       nsmp)