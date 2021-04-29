using Clustering
using Distributions: MvNormal, Normal, cdf
using Statistics: quantile
using Random: seed!
using LinearAlgebra


"""
     get_interpolators(X::Array{Float64,2},
                       assignments::Array{Int,1},
                       n_interpolators::Int)

Takes a Voronoi quantization as input, and gets 
additional local coordinates in stochastic space
later used for local interpolation.

"""
function get_interpolators(X::Array{Float64,2},
                           assignments::Array{Int,1},
                           centroids::Array{Float64,2},
                           n_wanted_interpolators::Int,
                           Λ::Array;
                           distance="L2-full")

                  
  nKL, nsmp_preconds = size(X)
  nsmp_preconds = length(assignments)
  nKL, P = size(centroids)
                              
  interpolators = Array{Array{Float64,2},1}(undef, P)
  n_interpolators = zeros(Int, P)

  for p in 1:P
    
    # assign p-th centroid as an interpolator
    interpolators[p] = centroids[:, p:p]
    n_interpolators[p] += 1

    # find points assigned to p-th cluster
    assigned_to_p = findall(x -> x == p, assignments)

    # if enough points assigned to p-th cluster for new clustering
    if length(assigned_to_p) >= n_wanted_interpolators - 1

      if distance == "L2-full"
        X[:, assigned_to_p] .*= sqrt.(Λ) 
        res = kmeans(X[:, assigned_to_p], n_wanted_interpolators - 1)
        X[:, assigned_to_p] ./= sqrt.(Λ)
        res.centers ./= sqrt.(Λ)

      elseif distance == "cdf-full"
        X[:, assigned_to_p] .= cdf.(Normal(), X[:, assigned_to_p]) 
        res = kmeans(X[:, assigned_to_p], n_wanted_interpolators - 1)
        X[:, assigned_to_p] .= quantile.(Normal(), X[:, assigned_to_p]) 
        res.centers .= quantile.(Normal(), res.centers)
        
      end
      interpolators[p] = hcat(interpolators[p], res.centers)
      n_interpolators[p] += n_wanted_interpolators - 1

    # otherwise, simply append remaining points
    else
      interpolators[p] = hcat(interpolators[p], X[:, assigned_to_p])
      n_interpolators[p] += size(X[:, assigned_to_p])[2]
    end

  end

  return interpolators, n_interpolators
end


struct InterpolatingPreconditioner
  coefficients::Array{Float64,1}
  Πs
end

"""
      apply_local_interpolation(Π::InterpolatingPreconditioner,
                                x::Array{Float64,1})

Applies local interpolation of preconditioners with given coefficients.

"""
function apply_local_interpolation(Π::InterpolatingPreconditioner,
                                   x::Array{Float64,1})


  Πx = zeros(Float64, length(x))

  for (i, ci) in enumerate(Π.coefficients)
    Πx .+= ci .* (Π.Πs[i] \ x)
  end

  return Πx

end

import Base: \
function (\)(Π::InterpolatingPreconditioner, x::Array{Float64,1})
  apply_local_interpolation(Π, x)
end

function LinearAlgebra.ldiv!(z::Array{Float64,1}, 
                             Π::InterpolatingPreconditioner,
                             r::Array{Float64,1})
  z .= apply_local_interpolation(Π, r)
end

function LinearAlgebra.ldiv!(Π::InterpolatingPreconditioner,
                             r::Array{Float64,1})
  r .= apply_local_interpolation(Π, copy(r))
end


"""
     preproc_interpolating_preconds(interpolators,
                                    n_interpolators,
                                    Λ, Ψ)
"""
function preproc_interpolating_preconds(interpolators::Array{Array{Float64,2},1},
                                        n_interpolators::Array{Int,1},
                                        Λ::Array{Float64,1},
                                        Ψ::Array{Float64,2})
  
  P = length(interpolators)
  nKL, _ = size(interpolators[1])

  sampler = prepare_mc_sampler(Λ, Ψ)

  Π_interpolators = []

  for p in 1:P

    Πs = []
     
    println("Gathering interpolators of preconditioner $p / $P")
    flush(stdout)
    for i in 1:n_interpolators[p]

      # assemble operator 
      set!(sampler, centroids[:, p])
      A, _ = do_isotropic_elliptic_assembly(cells, points,
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          point_markers,
                                          exp.(sampler.g), f, uexact)
        
      # build preconditioner
      println("prepare amg preconditioner $i / $(n_interpolators[p]) ...")
      flush(stdout)

      Π_amg = @time AMGPreconditioner{SmoothedAggregation}(A);

      # add preconditioner to list of interpolators
      push!(Πs, Π_amg)
    end

    push!(Π_interpolators, Πs)
  end
  
  return Π_interpolators
end 


"""
     shepard_interpolating_precond(ξ,
                                   n,
                                   local_interpolators, 
                                   Π_interpolators,
                                   distance="L2-full")

"""
function shepard_interpolating_precond(ξ::Array{Float64,1},
                                       n::Int,
                                       interpolators::Array{Float64,2}, 
                                       Π_interpolators;
                                       distance="L2-full")

  nKL, n_interpolators = size(interpolators)
  coefficients = Array{Float64,1}(undef, n_interpolators)
  
  distances_to_interpolators = Array{Float64,1}(undef, n_interpolators)
  Δξ = Array{Float64,1}(undef, nKL)

  W = 0.
  for i in 1:n_interpolators

    if distance == "L2-full"
      Δξ .= (interpolators[:, i] .- ξ) .* Λ
    elseif distance == "cdf-full"
      Δξ .= cdf.(Normal(), interpolators[:, i] .- ξ)
    end

    distances_to_interpolators[i] = sqrt(Δξ'Δξ)
    W += 1. / distances_to_interpolators[i]^2
  end

  for i in 1:n_interpolators
    coefficients[i] = (1. / distances_to_interpolators[i]^2) / W
  end

  Π = InterpolatingPreconditioner(coefficients, Π_interpolators)

  return Π
end



"""
     test_solver_with_interpolating_preconds()

"""
function test_solver_with_interpolating_preconds(nsmp,
                                                 Π_interpolators,
                                                 interpolators::Array{Array{Float64,2},1},
                                                 centroids::Array{Float64,2},
                                                 Λ::Array{Float64,1},
                                                 Ψ::Array{Float64,2};
                                                 maxit=5_000)

  nKL, P = size(centroids)
  sampler = prepare_mc_sampler(Λ, Ψ)

  seed!(123_456_789)
  X = rand(MvNormal(nKL, 1.), nsmp)

  assignments = zeros(Int, nsmp)
  iters = zeros(Int, nsmp)
  dists = zeros(Float64, nsmp)
  dists_to_0 = zeros(Float64, nsmp)

  Δx = Array{Float64,1}(undef, nKL)
  dist = Array{Float64,1}(undef, P)

  for ismp in 1:nsmp

    # assemble operator 
    set!(sampler, X[:, ismp])
    A, b = do_isotropic_elliptic_assembly(cells, points,
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          point_markers,
                                          exp.(sampler.g), f, uexact)  

    # find closest centroid
    for p in 1:P
      Δx = (centroids[:, p] .- X[:, ismp]) .* sqrt.(Λ)
      dist[p] = sqrt(Δx'Δx)
    end
    assignments[ismp] = argmin(dist)
    dists[ismp] = dist[assignments[ismp]]
    Δx = X[:, ismp] .* sqrt.(Λ)
    dists_to_0[ismp] = sqrt(Δx'Δx)                                          

    # Naively adapt locally interpolating preconditioner
    Π = shepard_interpolating_precond(X[:, ismp],
                                    A.n,
                                    interpolators[assignments[ismp]],
                                    Π_interpolators[assignments[ismp]],
                                    distance=distance)

    # solve system with closest constant preconditioner
    x = zeros(Float64, A.n)
    #_, iters[ismp], _  = pcg(A, b, x, Π[assignments[ismp]], maxit=maxit)
    _, iters[ismp], _  = pcg(A, b, x, Π, maxit=maxit)

    println("done solving system $ismp / $(nsmp) with $(iters[ismp]) iters. these")

  end # for ismp in 1:nsmp
end





function test_solver_with_centroidal_preconds(nsmp,
                                              Π,
                                              centroids::Array{Float64,2},
                                              Λ::Array{Float64,1},
                                              Ψ::Array{Float64,2};
                                              maxit=5_000)

  nKL, P = size(centroids)
  sampler = prepare_mc_sampler(Λ, Ψ)

  seed!(123_456_789)
  X = rand(MvNormal(nKL, 1.), nsmp)

  assignments = zeros(Int, nsmp)
  iters = zeros(Int, nsmp)
  dists = zeros(Float64, nsmp)
  dists_to_0 = zeros(Float64, nsmp)

  Δx = Array{Float64,1}(undef, nKL)
  dist = Array{Float64,1}(undef, P)

  for ismp in 1:nsmp

    # assemble operator 
    set!(sampler, X[:, ismp])
    A, b = do_isotropic_elliptic_assembly(cells, points,
                                          dirichlet_inds_g2l,
                                          not_dirichlet_inds_g2l,
                                          point_markers,
                                          exp.(sampler.g), f, uexact)  

    # find closest centroid
    for p in 1:P
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