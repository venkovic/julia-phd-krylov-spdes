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

function get_deterministic_grid(P, nKL, Λ, s)
  hξt_kmeans = Array{Float64,2}(undef, nKL, P)
  p = 0  
  hξt_kmeans[:, p+1] .= 0.
  if nKL > 0
    for p in 1:2^nKL
      bits = bitstring(p)[end-(nKL-1):end]
      for k in 1:nKL
        if bits[k] == '0'
          hξt_kmeans[k, p+1] = -s
        else
          hξt_kmeans[k, p+1] = s
        end
      end
    end
  end
  hηt_kmeans = sqrt.(Λ[1:nKL]) .* hξt_kmeans
  return hηt_kmeans, hξt_kmeans
end


function test_preconds()
  ξ, g = draw(Λ, Ψ)

  Threads.@threads for p in 1:P
    ξ[1:nKL_trunc] = hξt_kmeans[:, p]
    ξ[nKL_trunc+1:end] .= 0
    set!(Λ, Ψ, ξ, g)

    printlnln("do_isotropic_elliptic_assembly for preconditioner p = $p / $P ...")
    hAt, _ = @time do_isotropic_elliptic_assembly(cells, points,
                                                  dirichlet_inds_g2l,                                                
                                                  not_dirichlet_inds_g2l,
                                                  point_markers,
                                                  exp.(g), f, uexact)

    M_amg = AMGPreconditioner{SmoothedAggregation}(hAt)
    M_chol = cholesky(hAt)

    if precond == "k-means"
      Ξ = npzread("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.reals.npz")
    elseif precond == "determistic-grid"
      Ξ = npzread("data/Example20.m$m.P$P.p$p.reals.npz")
    end
    _, nreals = size(Ξ)
  
    iters_amg = Array{Int,1}(undef, nreals)
    iters_chol = Array{Int,1}(undef, nreals)

    for ireal in 1:nreals
      set!(Λ, Ψ, Ξ[:, ireal], g)

      printlnln("working on realization $ireal / $nreals for p = $p / $P ...")
      A, b = @time do_isotropic_elliptic_assembly(cells, points,
                                                  dirichlet_inds_g2l,
                                                  not_dirichlet_inds_g2l,
                                                  point_markers,
                                                  exp.(g), f, uexact)
                                                
      _, iters_amg[ireal], _ = @time pcg(A, b, zeros(A.n), M_amg)
    
      _, iters_chol[ireal], _ = @time pcg(A, b, zeros(A.n), M_chol)
    end
  
    if precond == "k-means"
      npzwrite("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.amg.iters.npz", iters_amg)
      npzwrite("data/Example20.dist$dist.nKL$nKL_trunc.P$P.p$p.chol.iters.npz", iters_chol)
    elseif precond == "deterministic-grid"
      npzwrite("data/Example20.m$m.P$P.p$p.amg.iters.npz", iters_amg)
      npzwrite("data/Example20.m$m.P$P.p$p.chol.iters.npz", iters_chol)
    end
  end
end





