using Clustering
using Distributions: MvNormal, Normal, cdf
using Random: seed!

"""
     get_quantizer(n::Int,
                   P::Int,
                   Λ::Array;
                   distance="L2-full")
Input:
 `n::Int`,
  number of of points from the stochastic space.

Output:
 `X::Array{Float64,2}`, `size(X) = (length(Λ), n)`,
  (non-weighted) coordinates in stochastic space.

 `centroids::Array{Float64,2}`, `size(centers) = (length(Λ), P)`,
  (non-weighted) coordinates of cluster centers in stochastic space.

 `assignments::Array{Int,1}`, `length(assignments) = n`,
  cluster to which each point in X is assigned.

 `costs::Array{Float64,1}`, `length(costs) = n`,
  cost of each point wrt to its center. 

"""
function get_quantizer(n::Int,
                       P::Int,
                       Λ::Array;
                       distance="L2-full")
  nKL = length(Λ)

  seed!(987_654_321)
  X = rand(MvNormal(nKL, 1.), n)

  if distance == "L2-full"
    X .*= sqrt.(Λ) 
    res = kmeans(X, P)
    X ./= sqrt.(Λ)
    res.centers ./= sqrt.(Λ)



  elseif distance == "cdf-full"
    X .= cdf 
    res = kmeans(X, P)
    X ./= sqrt.(Λ)
    res.centers ./= sqrt.(Λ)





  end

  return X, res.centers, res.assignments, res.costs
end


"""
     get_centroidal_preconds(preconds::Array{String,1},
                             centroids::Array{Float64,2},
                             Λ::Array{Float64,1},
                             Ψ::Array{Float64,2})

Prepares centroidal preconditioners of Voronoi quantizer.

Input:
 `centroids::Array{Float64,2}`,
  (non-weighted) centroidal coordinates in stochastic space.

 `Λ::Array{Float64,1}`,
  eigenvalues of KL expansion.

 `Ψ::Array{Float64,2}`,
  nodal values of eigevectors of KL expansion.

Output:
 `Π`,
  list of constant AMG preconditioners

"""
function get_centroidal_preconds(centroids::Array{Float64,2},
                                 Λ::Array{Float64,1},
                                 Ψ::Array{Float64,2})

  nKL, ncentroids = size(centroids)
  sampler = prepare_mc_sampler(Λ, Ψ)

  Π = []

  for p in 1:ncentroids

    # assemble operator 
    set!(sampler, centroids[:, p])
    A, _ = do_isotropic_elliptic_assembly(cells, points,
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          point_markers,
                                          exp.(sampler.g), f, uexact)
        
    # build preconditioner
    println("prepare amg preconditioner of centroid $p / $(ncentroids) ...")
    Π_amg = @time AMGPreconditioner{SmoothedAggregation}(A);
    push!(Π, Π_amg)
    flush(stdout)

  end
  
  return Π
end # function get_centroidal_preconds


function test_solver_with_centroidal_preconds(nsmp,
                                              Π,
                                              centroids::Array{Float64,2},
                                              Λ::Array{Float64,1},
                                              Ψ::Array{Float64,2};
                                              maxit=5_000)

  nKL, ncentroids = size(centroids)
  sampler = prepare_mc_sampler(Λ, Ψ)

  seed!(123_456_789)
  X = rand(MvNormal(nKL, 1.), nsmp)

  assignments = zeros(Int, nsmp)
  iters = zeros(Int, nsmp)
  dists = zeros(Float64, nsmp)
  dists_to_0 = zeros(Float64, nsmp)

  Δx = Array{Float64,1}(undef, nKL)
  dist = Array{Float64,1}(undef, ncentroids)

  for ismp in 1:nsmp
  
    # assemble operator 
    set!(sampler, X[:, ismp])
    A, b = do_isotropic_elliptic_assembly(cells, points,
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          point_markers,
                                          exp.(sampler.g), f, uexact)  

    # find closest centroid
    for p in 1:ncentroids
      Δx = (centroids[:, p] .- X[:, ismp]) .* sqrt.(Λ)
      dist[p] = sqrt(Δx'Δx)
    end
    assignments[ismp] = argmin(dist)
    dists[ismp] = dist[assignments[ismp]]
    Δx = X[:, ismp] .* sqrt.(Λ)
    dists_to_0[ismp] = sqrt(Δx'Δx)


    # solve system with closest constant preconditioner
    x = zeros(Float64, A.n)
    _, iters[ismp], _  = pcg(A, b, x, Π[assignments[ismp]], maxit=maxit)

    println("done solving system $ismp / $(nsmp) with $(iters[ismp]) iters.")

  end # for ismp in 1:nsmp

  return assignments, iters, dists, dists_to_0
end

