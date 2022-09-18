using Distributions: MvNormal, Normal, cdf
using Arpack: eigs

function get_data(ns::Int,
                  Λ::Array,
                  M::SparseArrays.SparseMatrixCSC{Float64,Int};
                  distance="L2")
  nKL = length(Λ)

  seed!(987_654_321)
  X = rand(MvNormal(nKL, 1.), ns)

  #chol = cholesky(M)
  #P = sparse(1:M.n, chol.p, ones(M.n))
  #L = P' * sparse(chol.L)
  # M ≈ L * L' 


  vals, vecs = eigs(M, nev=Int(.1 * M.n))
  sqrtM = vecs *  * vecs'

  if distance == "L2"
    X .*= sqrt.(Λ) 
    X .= L * X

  elseif distance == "cdf"
    X .= cdf.(Normal(), X)
    X .= L * X
    #X .= quantile.(Normal(), X)
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





function do_kmeans(ns, Ps, distances, nKL, M)
  hηt_kmeans = Dict(P => Dict(dist => Dict(nKL => zeros(nKL, P)) for dist in distances) for P in Ps)
  hξt_kmeans = Dict(P => Dict(dist => Dict(nKL => zeros(nKL, P)) for dist in distances) for P in Ps)
  
  get_ξ = true
  for P in Ps
    for dist in distances
      for nKL in nKLs
        for ireal in 1:nreals
          Λtrunc = Λ[1:nKL]
  
          X = get_data(ns, Λtrunc, M, distance=dist)
  
          println("kmeans working on ireal = $ireal, nKL = $nKL, dist = $dist, P = $P.")
          dt_kmeans[P][dist][nKL][ireal] = @elapsed res = kmeans(X, P)
          hηt_kmeans[P][dist][nKL] = res.centers
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