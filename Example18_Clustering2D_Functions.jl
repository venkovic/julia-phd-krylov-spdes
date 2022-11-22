using Distributions: MvNormal, Normal, cdf
using Arpack: eigs
using Random

function get_data(ns::Int,
                  Λ::Array;
                  distance="L2")
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

function do_kmeans(ns, Ps, distances, nKL, Λs)
  hηt_kmeans = Dict(P => Dict(dist => Dict(Λ => zeros(nKL, P) for Λ in Λs) for dist in distances) for P in Ps)
  hξt_kmeans = Dict(P => Dict(dist => Dict(Λ => zeros(nKL, P) for Λ in Λs) for dist in distances) for P in Ps)
  freqs_kmeans = Dict(P => Dict(dist => Dict(Λ => [0. for _ in 1:P] for Λ in Λs) for dist in distances) for P in Ps)
  w2_kmeans = Dict(P => Dict(dist => Dict(Λ => 0. for Λ in Λs) for dist in distances) for P in Ps)



  Random.seed!(3)

  for P in Ps
    for dist in distances
      for Λ in Λs
        Λtrunc = Λ[1:nKL]
        X = get_data(ns, Λtrunc, distance=dist)
        println("kmeans working on nKL = $nKL, dist = $dist, P = $P.")
        res = kmeans(X, P, tol=1e-8, maxiter=1_000)
        hηt_kmeans[P][dist][Λ] = res.centers
        hξt_kmeans[P][dist][Λ] = copy(hηt_kmeans[P][dist][Λ])
        freqs_kmeans[P][dist][Λ] = [sum(res.assignments .== p) / ns for p in 1:P]
        if dist == "L2"
          hξt_kmeans[P][dist][Λ] ./= sqrt.(Λtrunc)
        elseif dist == "cdf"
          hξt_kmeans[P][dist][Λ] ./= sqrt.(Λtrunc)
          hξt_kmeans[P][dist][Λ] .= quantile.(Normal(), hξt_kmeans[P][dist][Λ])
        end
        w2_kmeans[P][dist][Λ] = get_distortion(X, res.centers)
      end
    end
  end
  return hηt_kmeans, hξt_kmeans, freqs_kmeans, w2_kmeans
end