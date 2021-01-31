using Distributions: MvNormal


struct McSampler
  n::Int
  m::Int
  
  Ξ::MvNormal{Float64}
  ξ::Array{Float64,1}

  Λ::Array{Float64,1}
  Ψ::Array{Float64,2}
  g::Array{Float64,1}
end


"""
     prepare_mc_sampler(Λ::Array{Float64,1}
                        Ψ::Array{Float64,2})
        
"""
function prepare_mc_sampler(Λ::Array{Float64,1},
                            Ψ::Array{Float64,2})

  n, m = size(Ψ)

  Ξ = MvNormal(m, 1)
  ξ = rand(Ξ)
  
  g = Λ[1] * ξ[1] * Ψ[:, 1]
  for k in 2:m
    g .+= Λ[k] * ξ[k] * Ψ[:, k]
  end

  return McSampler(n, m,
                   Ξ, ξ,
                   Λ, Ψ, g)
end


"""
     function draw!(smp::McSampler)

"""
function draw!(smp::McSampler)
  smp.ξ .= rand(smp.Ξ)
  smp.g .= smp.Λ[1] * smp.ξ[1] * smp.Ψ[:, 1]
  for k in 2:smp.m
    smp.g .+= smp.Λ[k] * smp.ξ[k] * smp.Ψ[:, k]
  end
end


struct McmcSampler
  n::Int
  m::Int

  σ2::Float64
  Ξ::MvNormal{Float64}
  ξ::Array{Float64,1}

  ϑ2::Float64
  ΔΧ::MvNormal{Float64}
  χ::Array{Float64,1}

  Λ::Array{Float64,1}
  Ψ::Array{Float64,2}
  g::Array{Float64,1}
end


"""
     prepare_mcmc_sampler(Λ::Array{Float64,1}
                          Ψ::Array{Float64,2})
        
"""
function prepare_mcmc_sampler(Λ::Array{Float64,1},
                              Ψ::Array{Float64,2})

  n, m = size(Ψ)

  σ2 = 1.
  Ξ = MvNormal(m, σ2)
  ξ = rand(Ξ)

  ϑ2 = 2.38 * σ2  
  ΔΧ = MvNormal(m, ϑ2)
  χ = ξ .+ rand(ΔΧ)
  
  g = Λ[1] * ξ[1] * Ψ[:, 1]
  for k in 2:m
    g .+= Λ[k] * ξ[k] * Ψ[:, k]
  end

  return McmcSampler(n, m,
                     σ2, Ξ, ξ,
                     ϑ2, ΔΧ, χ,
                     Λ, Ψ, g)
end


"""
     function draw!(smp::McmcSampler)

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
  
  smp.g .= smp.Λ[1] * smp.χ[1] * smp.Ψ[:, 1]
  for k in 2:smp.m
    smp.g .+= smp.Λ[k] * smp.χ[k] * smp.Ψ[:, k]
  end

  return cnt
end