using Distributions: MvNormal, Normal, cdf

function get_data(ns::Int,
                  Λ::Array;
                  distance="L2-full")
  nKL = length(Λ)

  seed!(987_654_321)
  X = rand(MvNormal(nKL, 1.), ns)

  if distance == "L2-full"
    X .*= sqrt.(Λ) 
    X ./= sqrt.(Λ)

  elseif distance == "cdf-full"
    X .= cdf.(Normal(), X) 
    X .= quantile.(Normal(), X)
  end

  return X
end

function initialize_quantizer(X::Array{Float64,2}, P::Int, method="random")
  n, ns = size(X)
  initial_codebook = Array{Float64,2}(undef, n, P)
  if method == "random"
    inds = Array{Int,1}(undef, P)
    inds[1] = rand(1:ns)
    for i in 2:P
      ind = rand(1:ns)
      if ind in inds[1:i-1]
        while ind in inds[1:i-1]
          ind = rand(1:ns)
        end
      end
      inds[i] = ind
    end
    for (i, ind) in enumerate(inds)
      initial_codebook[:, i] .= X[:, ind]
    end
  end
  return initial_codebook
end

function get_gain_sequence(γ0::Float64, α::Float64, β::Float64, c::Float64, ns::Int)
  γ = Array{Float64,1}(undef, ns)
  for t in 1:ns
    γ[t] = γ0 * α / (t^c + β)
  end
  return γ
end

function get_distortion(X::Array{Float64,2}, hXt::Array{Float64,2})
  n, ns = size(X)
  n, P = size(hXt)
  w2 = 0.
  for s in 1:ns
    δw2_min = (X[:, s] - hXt[:, 1])'*(X[:, s] - hXt[:, 1])
    for p in 2:P
      δw2_p = (X[:, s] - hXt[:, p])'*(X[:, s] - hXt[:, p])
      if δw2_p < δw2_min
        δw2_min = δw2_p
      end
    end
    w2 += δw2_min
  end
  w2 /= ns
  return w2
end

function clvq(X::Array{Float64,2}, P::Int, γ::Array{Float64,1})
  n, ns = size(X)
  hXt = initialize_quantizer(X, P)
  for t in 0:ns-1
    #
    # Competitive phase: Find nearest centroid
    MinSquareDist = (X[:, t+1] - hXt[:, 1])'*(X[:, t+1] - hXt[:, 1])
    pMin = 1
    for p in 2:P
      SquareDist = (X[:, t+1] - hXt[:, p])'*(X[:, t+1] - hXt[:, p])
      if SquareDist < MinSquareDist
        MinSquareDist = SquareDist
        pMin = p
      end
    end
    #
    # Learning phase
    hXt[:, pMin] .-= γ[t+1] * (hXt[:, pMin] .- X[:, t+1])
  end

  return hXt
end

function test_clvq(ns, Ps, distances, nKLs, nreals)
  dt_clvq = Dict(P => Dict(dist => Dict(nKL => zeros(nreals) for nKL in nKLs) for dist in distances) for P in Ps)
  w2_clvq = Dict(P => Dict(dist => Dict(nKL => zeros(nreals) for nKL in nKLs) for dist in distances) for P in Ps)
  
  get_ξ = false
  for P in Ps
    for dist in distances
      for nKL in nKLs
        for ireal in 1:nreals
          Λtrunc = Λ[1:nKL]
    
          X = get_data(ns, Λtrunc, distance=dist)
    
          γ = get_gain_sequence(1., .1, .2, .51, ns)
    
          println("clvq working on ireal = $ireal, nKL = $nKL, dist = $dist, P = $P.")
          dt_clvq[P][dist][nKL][ireal]  = @elapsed hηt_clvq = clvq(X, P, γ)
          w2_clvq[P][dist][nKL][ireal] = get_distortion(X, hηt_clvq)
          if get_ξ
            hξt_clvq = copy(hηt_clvq)
            if dist == "L2"
              hξt_clvq ./= sqrt.(Λtrunc)
            elseif dist == "cdf"
              hξt_clvq .= quantile.(Normal(), hξt_clvq)
            end
          end
        end
      end
    end
  end
  
  return dt_clvq, w2_clvq
end
  

function test_kmeans(ns, Ps, distances, nKLs, nreals)
  dt_kmeans = Dict(P => Dict(dist => Dict(nKL => zeros(nreals) for nKL in nKLs) for dist in distances) for P in Ps)
  w2_kmeans = Dict(P => Dict(dist => Dict(nKL => zeros(nreals) for nKL in nKLs) for dist in distances) for P in Ps)
  
  get_ξ = false
  for P in Ps
    for dist in distances
      for nKL in nKLs
        for ireal in 1:nreals
          Λtrunc = Λ[1:nKL]
  
          X = get_data(ns, Λtrunc, distance=dist)
  
          println("kmeans working on ireal = $ireal, nKL = $nKL, dist = $dist, P = $P.")
          dt_kmeans[P][dist][nKL][ireal] = @elapsed res = kmeans(X, P)
          hηt_kmeans = res.centers
          w2_kmeans[P][dist][nKL][ireal] = get_distortion(X, hηt_kmeans)
          if get_ξ
            hξt_kmeans = copy(hηt_kmeans)
            if dist == "L2"
              hξt_kmeans ./= sqrt.(Λtrunc)
            elseif dist == "cdf"
              hξt_kmeans .= quantile.(Normal(), hξt_kmeans)
            end
          end
        end
      end
    end
  end
  return dt_kmeans, w2_kmeans
end