using TriangleMesh
push!(LOAD_PATH, "./Fem/")
using Fem
using Arpack
using Distributions
using NPZ

poly = polygon_Lshape()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

L = .1
sig2 = 1.

function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  return sig2 * exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end
  
C = @time do_mass_covariance_assembly(mesh.cell, mesh.point, cov)
M = @time get_mass_matrix(mesh.cell, mesh.point)

λ, Φ = map(x -> real(x), eigs(C, M, nev=400))

function draw(λ, Φ)
  n, nmode = size(Φ)
  ξ = rand(Normal(), nmode)
  g = zeros(n)
  for k in 1:nmode
    g .+= sqrt(λ[k]) * ξ[k] * Φ[:, k]
  end
  return ξ, g
end

area = get_total_area(mesh.cell, mesh.point)

ξ, g = draw(λ, Φ)

@time npzwrite("cells.npz", mesh.cell' .- 1)
@time npzwrite("points.npz", mesh.point')
@time npzwrite("g.npz", g)