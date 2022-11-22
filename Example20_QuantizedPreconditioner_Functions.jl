using Distributions: MvNormal, Normal, cdf, quantile
using Random
using Clustering: kmeans

function get_scaled_data(ns::Int,
                         Λ::Vector{Float64},
                         distance::String)
  nKL = length(Λ)

  seed!(987_654_321)
  X = rand(MvNormal(nKL, 1.), ns)

  if distance == "L2"
    X .*= sqrt.(Λ) 

  elseif distance == "cdf"
    X .= cdf.(Normal(), X)
    X .*= sqrt.(Λ)
  end

  return X
end

function scale_data(ξ::Vector{Float64},
                    Λ::Vector{Float64},
                    distance::String)
  
  if distance == "L2"
    ξ .*= sqrt.(Λ)

  elseif distance == "cdf"
    ξ .= cdf.(Normal(), ξ)
    ξ .*= sqrt.(Λ)
  end

  return ξ
end

function find_precond(x::Vector{Float64}, hXt::Array{Float64,2})
  nKL_trunc, P = size(hXt)
  nKL = length(x)
  hXt_padded = zeros(nKL, P)
  hXt_padded[1:nKL_trunc, 1:P] = hXt
  min_dist = (x .- hXt_padded[:, 1])' * (x .- hXt_padded[:, 1])
  p_min_dist = 1
  for p in 2:P
    dist = (x .- hXt_padded[:, p])' * (x .- hXt_padded[:, p])
    if dist < min_dist
      min_dist = dist
      p_min_dist = p
    end
  end
  return p_min_dist
end

function do_kmeans(ns, P, dist, nKL, Λ)
    hηt_kmeans = zeros(nKL, P)
    hξt_kmeans = zeros(nKL, P)

    Λtrunc = Λ[1:nKL]
    X = get_scaled_data(ns, Λtrunc, dist)
    println("kmeans working on nKL = $nKL, dist = $dist.")
    res = kmeans(X, P, tol=1e-8, maxiter=1_000)
    hηt_kmeans = res.centers
    hξt_kmeans = copy(hηt_kmeans)
    if dist == "L2"
      hξt_kmeans ./= sqrt.(Λtrunc)
    elseif dist == "cdf"
      hξt_kmeans ./= sqrt.(Λtrunc)
      hξt_kmeans .= quantile.(Normal(), hξt_kmeans)
    end
    return hηt_kmeans, hξt_kmeans
  end

function get_deterministic_grid()
  nothing  
end