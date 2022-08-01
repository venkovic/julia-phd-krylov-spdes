struct McSampler
  n::Int # number of DoFs
  m::Int # number of modes in Karhunen Loeve (KL) expansion
  
  Ξ::MvNormal{Float64} # multvariate latent random vector (rv)
  ξ::Array{Float64,1} # realization of rv

  Λ::Array{Float64,1} # dominant eigenvalues of KL expansion
  Ψ::Array{Float64,2} # eigenvectors of KL expansion stored by columns
  g::Array{Float64,1} # realization
end


"""
     prepare_mc_sampler(Λ::Array{Float64,1},
                        Ψ::Array{Float64,2})

Prepares and returns instance of McSampler.
  
Input:

 `Λ::Array{Float64,1}`,
  dominant eigenvalues of Karhunen Loeve (KL) expansion.

 `Ψ::Array{Float64,2}`, 
  eigenvectors of KL expansion stored by columns.

Output:

 Instance of `McSampler`.

"""
function prepare_mc_sampler(Λ::Array{Float64,1},
                            Ψ::Array{Float64,2})

  n, m = size(Ψ)

  Ξ = MvNormal(m, 1.)
  ξ = rand(Ξ)
  
  g = sqrt(Λ[1]) * ξ[1] * Ψ[:, 1]
  for k in 2:m
    g .+= sqrt(Λ[k]) * ξ[k] * Ψ[:, k]
  end

  return McSampler(n, m,
                   Ξ, ξ,
                   Λ, Ψ, g)
end


"""
     function draw!(smp::McSampler)

Draws a realization of McSampler in place of smp.g.

Input:

 `smp::McSampler`

"""
function draw!(smp::McSampler)
  smp.ξ .= rand(smp.Ξ)
  smp.g .= sqrt(smp.Λ[1]) * smp.ξ[1] * smp.Ψ[:, 1]
  for k in 2:smp.m
    smp.g .+= sqrt(smp.Λ[k]) * smp.ξ[k] * smp.Ψ[:, k]
  end
end


"""
     function set!(smp::McSampler, ξ)

Forcess a realization of McSampler in place of smp.g for a given ξ.

Input:

 `smp::McSampler`

"""
function set!(smp::McSampler, ξ::Array{Float64,1})
  smp.ξ .= ξ
  smp.g .= sqrt(smp.Λ[1]) * smp.ξ[1] * smp.Ψ[:, 1]
  for k in 2:smp.m
    smp.g .+= sqrt(smp.Λ[k]) * smp.ξ[k] * smp.Ψ[:, k]
  end
end





struct McmcSampler
  n::Int # number of DoFs
  m::Int # number of modes in Karhunen Loeve (KL) expansion

  σ2::Float64 # variance
  Ξ::MvNormal{Float64} # multvariate latent random vector (rv)
  ξ::Array{Float64,1} # realization of rv

  ϑ2::Float64 # variance
  ΔΧ::MvNormal{Float64} # multvariate latent rv
  χ::Array{Float64,1} # realization of rv

  Λ::Array{Float64,1} # dominant eigenvalues of KL expansion
  Ψ::Array{Float64,2} # eigenvectors of KL expansion stored by columns
  g::Array{Float64,1} # realization
end


"""
     prepare_mcmc_sampler(Λ::Array{Float64,1}
                          Ψ::Array{Float64,2})

Prepares and returns instance of McmcSampler.
  
Input:
                          
 `Λ::Array{Float64,1}`,
  dominant eigenvalues of Karhunen Loeve (KL) expansion.
                          
 `Ψ::Array{Float64,2}`, 
  eigenvectors of KL expansion stored by columns.
                          
Output:
                          
  Instance of `McmcSampler`.

"""
function prepare_mcmc_sampler(Λ::Array{Float64,1},
                              Ψ::Array{Float64,2})

  n, m = size(Ψ)

  σ2 = 1.
  Ξ = MvNormal(m, sqrt(σ2))
  ξ = rand(Ξ)

  ϑ2 = 2.38^2 * σ2 / m
  ΔΧ = MvNormal(m, sqrt(ϑ2))
  χ = ξ .+ rand(ΔΧ)
  
  g = sqrt(Λ[1]) * ξ[1] * Ψ[:, 1]
  for k in 2:m
    g .+= sqrt(Λ[k]) * ξ[k] * Ψ[:, k]
  end

  return McmcSampler(n, m,
                     σ2, Ξ, ξ,
                     ϑ2, ΔΧ, χ,
                     Λ, Ψ, g)
end


"""
     function draw!(smp::McmcSampler)

Draws a realization of McmcSampler in place of smp.g.

Input:

 `smp::McmcSampler`

"""
function draw!(smp::McmcSampler)
  
  sq_norm_of_ξ = smp.ξ'smp.ξ
  
  smp.χ .= smp.ξ .+ rand(smp.ΔΧ)
  sq_norm_of_χ = smp.χ'smp.χ
  cnt = 1

  if sq_norm_of_ξ < sq_norm_of_χ
    α = exp((sq_norm_of_ξ - sq_norm_of_χ) / 2 / smp.σ2)
  else
    α = 1.
  end
  u = rand()
  
  while u > α
    smp.χ .= smp.ξ .+ rand(smp.ΔΧ)
    sq_norm_of_χ = smp.χ'smp.χ
    if sq_norm_of_ξ < sq_norm_of_χ
      α = exp((sq_norm_of_ξ - sq_norm_of_χ) / 2 / smp.σ2)
    else
      α = 1.
    end
    u = rand()
    cnt += 1
  end
  smp.ξ .= smp.χ
  
  smp.g .= sqrt(smp.Λ[1]) * smp.ξ[1] * smp.Ψ[:, 1]
  for k in 2:smp.m
    smp.g .+= sqrt(smp.Λ[k]) * smp.ξ[k] * smp.Ψ[:, k]
  end

  return cnt
end








struct HybridSampler
  n::Int # number of DoFs
  m::Int # number of modes in Karhunen Loeve (KL) expansion
  m_mcmc::Int # number of modes in KL expansion sampled by MCMC

  σ2::Float64 # variance
  Ξ::MvNormal{Float64} # multvariate latent random vector (rv)
  ξ::Array{Float64,1} # realization of rv

  ϑ2::Float64 # variance
  ΔΧ::MvNormal{Float64} # multvariate latent rv
  χ::Array{Float64,1} # realization of rv

  Λ::Array{Float64,1} # dominant eigenvalues of KL expansion
  Ψ::Array{Float64,2} # eigenvectors of KL expansion stored by columns
  g::Array{Float64,1} # realization
end


"""
     prepare_hybrid_sampler(Λ::Array{Float64,1}
                            Ψ::Array{Float64,2},
                            m_mcmc::Int)

Prepares and returns instance of HybridSampler.
  
Input:
                          
 `Λ::Array{Float64,1}`,
  dominant eigenvalues of Karhunen Loeve (KL) expansion.
                          
 `Ψ::Array{Float64,2}`, 
  eigenvectors of KL expansion stored by columns.

 `m_mcmc::Int`,
  number of modes in KL expansion sampled by MCMC.
Output:
                          
  Instance of `HybridSampler`.

"""
function prepare_hybrid_sampler(Λ::Array{Float64,1},
                                Ψ::Array{Float64,2},
                                m_mcmc::Int)

  n, m = size(Ψ)
  ξ = Array{Float64}(undef, m)

  σ2 = 1.
  Ξ = MvNormal(m_mcmc, sqrt(σ2))
  ξ[1:m_mcmc] = rand(Ξ)
  ξ[m_mcmc+1:m] = rand(MvNormal(m-m_mcmc, 1.))

  ϑ2 = 2.38^2 * σ2 / m_mcmc
  ΔΧ = MvNormal(m_mcmc, sqrt(ϑ2))
  χ = ξ[1:m_mcmc] .+ rand(ΔΧ)
  
  g = sqrt(Λ[1]) * ξ[1] * Ψ[:, 1]
  for k in 2:m
    g .+= sqrt(Λ[k]) * ξ[k] * Ψ[:, k]
  end

  return HybridSampler(n, m, m_mcmc,
                       σ2, Ξ, ξ,
                       ϑ2, ΔΧ, χ,
                       Λ, Ψ, g)
end


"""
     function draw!(smp::HybridSampler)

Draws a realization of HybridSampler in place of smp.g.

Input:

 `smp::HybridSampler`

"""
function draw!(smp::HybridSampler)
  
  sq_norm_of_ξ = smp.ξ[1:smp.n_mcmc]'smp.ξ[1:smp.n_mcmc]
  
  smp.χ .= smp.ξ[1:smp.n_mcmc] .+ rand(smp.ΔΧ)
  sq_norm_of_χ = smp.χ'smp.χ
  cnt = 1

  if sq_norm_of_ξ < sq_norm_of_χ
    α = exp((sq_norm_of_ξ - sq_norm_of_χ) / 2 / smp.σ2)
  else
    α = 1.
  end
  u = rand()
  
  while u > α
    smp.χ .= smp.ξ[1:smp.n_mcmc] .+ rand(smp.ΔΧ)
    sq_norm_of_χ = smp.χ'smp.χ
    if sq_norm_of_ξ < sq_norm_of_χ
      α = exp((sq_norm_of_ξ - sq_norm_of_χ) / 2 / smp.σ2)
    else
      α = 1.
    end
    u = rand()
    cnt += 1
  end
  smp.ξ[1:smp.n_mcmc] .= smp.χ
  smp.ξ[m_mcmc+1:m] = rand(MvNormal(smp.m-smp.m_mcmc, 1.))

  smp.g .= sqrt(smp.Λ[1]) * smp.ξ[1] * smp.Ψ[:, 1]
  for k in 2:smp.m
    smp.g .+= sqrt(smp.Λ[k]) * smp.ξ[k] * smp.Ψ[:, k]
  end

  return cnt
end




























