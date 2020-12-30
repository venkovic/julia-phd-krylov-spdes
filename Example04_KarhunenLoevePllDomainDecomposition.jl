#using Distributed
#addprocs(3)
#addprocs(([("marcel", 4)]), tunnel=true)
#addprocs(([("andrew", 4)]), tunnel=true)

push!(LOAD_PATH, "./Fem/")
using Fem
using TriangleMesh
using NPZ
using Arpack

poly = polygon_unitSquare()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

ndom = 40
epart, npart = mesh_partition(mesh, ndom)

L = .1
sig2 = 1.

function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

for idom in 1:ndom
  inds_l2g, inds_g2l, elems = set_subdomain(mesh, epart, idom)
  C = do_local_mass_covariance_assembly(mesh.cell, mesh.point, inds_l2g, inds_g2l, elems, cov)
  M = do_local_mass_assembly(mesh.cell, mesh.point, inds_g2l, elems)
  λ, Φ = map(x -> real(x), eigs(C, M))
  println("$idom, $(length(inds_l2g)), $(sum(λ))")
end
