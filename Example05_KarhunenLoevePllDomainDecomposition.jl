using Distributed
addprocs(3)
#addprocs(([("marcel", 4)]), tunnel=true)
#addprocs(([("andrew", 4)]), tunnel=true)

@everywhere push!(LOAD_PATH, "./Fem/")
@everywhere using Fem

using TriangleMesh
using NPZ
import LinearAlgebra
import Arpack
using Distributions

poly = polygon_unitSquare()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, 
                   set_area_max=true)

@everywhere ndom = 20
nev = 35
epart, npart = mesh_partition(mesh, ndom)

@everywhere L = .1
@everywhere sig2 = 1.

@everywhere function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

inds_g2ld = [Dict{Int,Int}() for _ in 1:ndom]
inds_l2gd = Array{Int,1}[]
elemsd = Array{Int,1}[]
ϕd = Array{Float64,2}[]

@distributed for idom in 1:ndom
  println("$idom, $(myid())")
  #inds_l2g, inds_g2l, elems = set_subdomain(mesh, epart, idom)
  #C = do_local_mass_covariance_assembly(mesh.cell, mesh.point, inds_l2g, inds_g2l, 
  #                                      elems, cov)
  #M = do_local_mass_assembly(mesh.cell, mesh.point, inds_g2l, elems)
  #λ, ϕ = map(x -> real(x), Arpack.eigs(C, M, nev=nev))
  #inds_g2ld[idom] = inds_g2l
  #push!(inds_l2gd, inds_l2g)
  #push!(elemsd, elems)
  #push!(ϕd, ϕ)
  #println("$idom, $(length(inds_l2g)), $(sum(λ))")
end
@sync nothing