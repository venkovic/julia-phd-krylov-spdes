using SparseArrays: SparseMatrixCSC
using LinearMaps: FunctionMap

"""
     get_constant_preconds()

Returns queried constant preconditioners.

"""
function get_constant_preconds(A_0::SparseMatrixCSC{Float64,Int})

  printlnln("prepare amg_0 preconditioner for A_0 ...")
  Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);
  flush(stdout)    
  return Π_amg_0

end

"""
     find_index_of_closest_vertex(x::Float64, 
                                  y::Float64,
                                  points::,
                                  not_dirichlet_inds_g2l::)

Returns index of closest vertex in the mesh.

"""
function find_index_of_closest_vertex(x::Float64,
                                      y::Float64,
                                      points::Array{Float64,2},
                                      not_dirichlet_inds_g2l::Dict{Int,Int})

  index, min_dist = 0, 1e12
  for (key, ind) in not_dirichlet_inds_g2l 
    dist = sqrt((points[1, key] - x)^2 + (points[2, key] - y)^2)
    if dist < min_dist
      index = ind
      min_dist = dist
    end
  end
  return index

end


"""
     solve_mc_sample(nsmp::Int,
                     Λ::Array{Float64,1},
                     Ψ::Array{Float64,2},
                     Π0,
                     maxit::Int,
                     index::Int;
                     verbose=true,)

"""
function solve_mc_sample(nsmp::Int,
                         Λ::Array{Float64,1},
                         Ψ::Array{Float64,2},
                         Π0,
                         maxit::Int,
                         index::Int;
                         verbose=true,)

  sampler = prepare_mc_sampler(Λ, Ψ)

  verbose ? println("\n1 / $nsmp") : nothing
  verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
  Δt = @elapsed A, b = do_isotropic_elliptic_assembly(cells, points,
                                                      dirichlet_inds_g2l,
                                                      not_dirichlet_inds_g2l,
                                                      point_markers,
                                                      exp.(sampler.g),
                                                      f, uexact)
  verbose ? println("$Δt seconds") : nothing

  iter = Array{Int,1}(undef, nsmp)
  x = zeros(Float64, A.n)
  u_values = Array{Float64,1}(undef, nsmp)

  #
  # Sample ξ by mc and solve linear systems by pcg
  #
  it = 0
  for s in 1:nsmp

    if s > 1
      verbose ? print("\n$s / $nsmp\n") : nothing
      draw!(sampler)

      verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
      Δt = @elapsed update_isotropic_elliptic_assembly!(A, b,
                                                        cells, points,
                                                        dirichlet_inds_g2l,
                                                        not_dirichlet_inds_g2l,
                                                        point_markers,
                                                        exp.(sampler.g),
                                                        f, uexact)
      verbose ? println("$Δt seconds") : nothing

    end

    x .= 0
    Δt = @elapsed _, it, _  = pcg(A, b, x, Π0, maxit=maxit)
    u_values[s] = x[index]
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[s] = it
    flush(stdout)

  end # for s in 1:nsmp

  return u_values
end



"""
     solve_single_chain(nsmp::Int,
                        Λ::Array{Float64,1},
                        Ψ::Array{Float64,2},
                        Π0,
                        maxit::Int,
                        index::Int;
                        verbose=true,)

"""
function solve_single_chain(nsmp::Int,
                            Λ::Array{Float64,1},
                            Ψ::Array{Float64,2},
                            n_mcmc::Int,
                            Π0,
                            maxit::Int,
                            index::Int;
                            verbose=true,)

  nKL = length(Λ)
  if n_mcmc <= nKL
    sampler = prepare_mcmc_sampler(Λ, Ψ)
  else
    sampler = prepare_hybrid_sampler(Λ, Ψ, n_mcmc)
  end

  verbose ? println("\n1 / $nsmp") : nothing
  verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
  Δt = @elapsed A, b = do_isotropic_elliptic_assembly(cells, points,
                                                      dirichlet_inds_g2l,
                                                      not_dirichlet_inds_g2l,
                                                      point_markers,
                                                      exp.(sampler.g),
                                                      f, uexact)
  verbose ? println("$Δt seconds") : nothing

  iter = Array{Int,1}(undef, nsmp)
  repeats = Array{Int,1}(undef, nsmp)
  x = zeros(Float64, A.n)
  u_values = Array{Float64,1}(undef, nsmp)

  #
  # Sample ξ by mcmc and solve linear systems by pcg
  #
  it = 0
  repeats[1] = 1
  cnt_reals = 1
  for s in 1:nsmp

    if s > 1
      verbose ? print("\n$s / $nsmp") : nothing
      repeats[s] = draw!(sampler)
      cnt_reals += repeats[s]
      verbose ? println(" (acceptance frequency: $(s / cnt_reals))") : nothing

      verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
      Δt = @elapsed update_isotropic_elliptic_assembly!(A, b,
                                                        cells, points,
                                                        dirichlet_inds_g2l,
                                                        not_dirichlet_inds_g2l,
                                                        point_markers,
                                                        exp.(sampler.g),
                                                        f, uexact)
      verbose ? println("$Δt seconds") : nothing

    end

    x .= 0
    Δt = @elapsed _, it, _  = pcg(A, b, x, Π0, maxit=maxit)
    u_values[s] = x[index]
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[s] = it
    flush(stdout)

  end # for s in 1:nsmp

  p_acc = nsmp / cnt_reals

  return u_values, repeats
end