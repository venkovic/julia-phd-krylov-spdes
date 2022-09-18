using SparseArrays: SparseMatrixCSC
using LinearMaps: FunctionMap

"""
     get_constant_preconds()

Returns queried constant preconditioners.

"""
function get_constant_precond(precond::String,
                                  A_0::SparseMatrixCSC{Float64,Int})

  precond in preconds_with_dd ? do_dd = true : do_dd = false

  if precond == "amg_0"
    printlnln("prepare amg_0 preconditioner for A_0 ...")
    Π = @time AMGPreconditioner{SmoothedAggregation}(A_0);
    flush(stdout)

  elseif precond == "lorasc$(ndom)_0"
    printlnln("prepare_lorasc_precond for A_0 with ε = 0 ...")
    Π = @time prepare_lorasc_precond(ndom,
                                     cells,
                                     points,
                                     epart,
                                     ind_Id_g2l,
                                     ind_Γd_g2l,
                                     ind_Γ_g2l,
                                     ind_Γd_Γ2l,
                                     node_owner,
                                     node_Γ_cnt,
                                     exp.(g_0),
                                     dirichlet_inds_g2l,
                                     not_dirichlet_inds_g2l,
                                     f,
                                     uexact,
                                     do_local_schur_assembly=true,
                                     low_rank_correction=:exact,
                                     ε=.01)
    flush(stdout)

  elseif precond == "lorasc$(ndom)_1"
    printlnln("prepare_lorasc_precond for A_0 with ε = 0.01 ...")
    Π = @time prepare_lorasc_precond(ndom,
                                     cells,
                                     points,
                                     epart,
                                     ind_Id_g2l,
                                     ind_Γd_g2l,
                                     ind_Γ_g2l,
                                     ind_Γd_Γ2l,
                                     node_owner,
                                     node_Γ_cnt,
                                     exp.(g_0),
                                     dirichlet_inds_g2l,
                                     not_dirichlet_inds_g2l,
                                     f,
                                     uexact,
                                     do_local_schur_assembly=true,
                                     low_rank_correction=:exact,
                                     ε=0)
    flush(stdout)

  elseif precond == "bj$(nbj)_0"
    printlnln("prepare bj_0 preconditioner for A_0 ...")
    Π = @time BJPreconditioner(nbj, A_0);
    flush(stdout)
  end

  return Π

end # function get_constant_preconds


"""
test_solvers_on_a_chain(nsmp::Int,
                        Λ::Array{Float64,1},
                        Ψ::Array{Float64,2}
                        Π,
                        precond::String,
                        nvec::Int,
                        spdim::Int
                        maxit::Int;
                        verbose=true)

"""
function test_solvers_on_single_chain(nsmp::Int,
                                      Λ::Array{Float64,1},
                                      Ψ::Array{Float64,2},
                                      Π,
                                      precond::String,
                                      nvec::Int,
                                      spdim::Int,
                                      maxit::Int;
                                      verbose=true)

  sampler = prepare_mcmc_sampler(Λ, Ψ)

  verbose ? println("\n1 / $nsmp") : nothing
  verbose ? print("do_isotropic_elliptic_assembly ... ") : nothing
  Δt = @elapsed A, b = do_isotropic_elliptic_assembly(cells, points,
                                                      dirichlet_inds_g2l,
                                                      not_dirichlet_inds_g2l,
                                                      point_markers,
                                                      exp.(sampler.g),
                                                      f, uexact)
  verbose ? println("$Δt seconds") : nothing


  methods = String[]
  W = Dict{String,Array{Float64,2}}()
  iter = Dict{String,Int}()

  method = precond * "-PCG"
  push!(methods, method)

  method = precond * "-RR-Def-PCG"
  push!(methods, method)
  W[method] = Array{Float64,2}(undef, A.n, nvec)

  method = precond * "-HR-Def-PCG"
  push!(methods, method)
  W[method] = Array{Float64,2}(undef, A.n, nvec)

  method = precond * "-TR-RR-Def-PCG"
  push!(methods, method)
  W[method] = Array{Float64,2}(undef, A.n, nvec)

  method = precond * "-TR-HR-Def-PCG"
  push!(methods, method)
  W[method] = Array{Float64,2}(undef, A.n, nvec)

  method = precond * "-LO-TR-RR-Def-PCG"
  push!(methods, method)
  W[method] = Array{Float64,2}(undef, A.n, nvec)
  
  method = precond * "-LO-TR-HR-Def-PCG"
  push!(methods, method)
  W[method] = Array{Float64,2}(undef, A.n, nvec)

  x = zeros(Float64, A.n)

  #
  # Sample ξ by mcmc and solve linear systems by def-pcg with 
  # online eigenvectors approximation
  #
  it = 0
  cnt_reals = 1
  for s in 1:nsmp

    if s > 1
      verbose ? print("\n$s / $nsmp") : nothing
      cnt_reals += draw!(sampler)
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

    end # if s > 1

    method = precond * "-PCG"
    verbose ? print("$method of A * u = b ... ") : nothing
    x .= 0
    Δt = @elapsed _, it, _ = pcg(A, b, x, Π, maxit=maxit)
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[method] = it
    flush(stdout)

    method = precond * "-RR-Def-PCG"
    verbose ? print("$method of A * u = b ... ") : nothing
    if s == 1
      x .= 0
      Δt = @elapsed _, it, _, W[method] = rrpcg(A, b, x, Π, nvec, spdim, maxit=maxit)
    else
      x .= 0
      Δt = @elapsed _, it, _, W[method] = rrdefpcg(A, b, x, W[method], Π, spdim, maxit=maxit)
    end
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[method] = it
    flush(stdout)

    method = precond * "-HR-Def-PCG"
    verbose ? print("$method of A * u = b ... ") : nothing
    if s == 1
      x .= 0
      Δt = @elapsed _, it, _, W[method] = hrpcg(A, b, x, Π, nvec, spdim, maxit=maxit)
    else
      x .= 0
      Δt = @elapsed _, it, _, W[method] = hrdefpcg(A, b, x, W[method], Π, spdim, maxit=maxit)
    end
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[method] = it
    flush(stdout)

    method = precond * "-TR-RR-Def-PCG"
    verbose ? print("$method of A * u = b ... ") : nothing
    if s == 1
      x .= 0
      Δt = @elapsed _, it, _, W[method] = trrrpcg(A, b, x, Π, nvec, spdim, maxit=maxit)
    else
      x .= 0
      Δt = @elapsed _, it, _, W[method] = trrrdefpcg(A, b, x, W[method], Π, spdim, maxit=maxit)
    end
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[method] = it
    flush(stdout)

    method = precond * "-TR-HR-Def-PCG"
    verbose ? print("$method of A * u = b ... ") : nothing
    if s == 1
      x .= 0
      Δt = @elapsed _, it, _, W[method] = trhrpcg(A, b, x, Π, nvec, spdim, maxit=maxit)
    else
      x .= 0
      Δt = @elapsed _, it, _, W[method] = trhrdefpcg(A, b, x, W[method], Π, spdim, maxit=maxit)
    end
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[method] = it
    flush(stdout)

    method = precond * "-LO-TR-RR-Def-PCG"
    verbose ? print("$method of A * u = b ... ") : nothing
    if s == 1
      x .= 0
      Δt = @elapsed _, it, _, W[method] = lotrrrpcg(A, b, x, Π, nvec, spdim, maxit=maxit)
    else
      x .= 0
      Δt = @elapsed _, it, _, W[method] = lotrrrdefpcg(A, b, x, W[method], Π, spdim, maxit=maxit)
    end
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[method] = it
    flush(stdout)

    method = precond * "-LO-TR-HR-Def-PCG"
    verbose ? print("$method of A * u = b ... ") : nothing
    if s == 1
      x .= 0
      Δt = @elapsed _, it, _, W[method] = lotrhrpcg(A, b, x, Π, nvec, spdim, maxit=maxit)
    else
      x .= 0
      Δt = @elapsed _, it, _, W[method] = lotrhrdefpcg(A, b, x, W[method], Π, spdim, maxit=maxit)
    end
    verbose ? println("$Δt seconds, iter = $it") : nothing
    iter[method] = it
    flush(stdout)

  end # for s in 1:nsmp

  return iter
end


"""
test_solvers_on_several_chains(nchains::Int,
                               nsmp::Int,
                               Λ::Array{Float64,1},
                               Ψ::Array{Float64,2},
                               Π,
                               preconds::String,
                               nvec::Int,
                               spdim::Int
                               maxit::Int;
                               verbose=true)

"""
function test_solvers_on_several_chains(nchains::Int,
                                        nsmp::Int,
                                        Λ::Array{Float64,1},
                                        Ψ::Array{Float64,2},
                                        Π,
                                        precond::String,
                                        nvec::Int,
                                        spdim::Int,
                                        maxit::Int;
                                        verbose=true)

  iters = Dict{String,Array{Int}}()

  for ichain in 1:nchains

    println("\n\nworking on chain $ichain / $nchains ...")
    
    iter = Dict{String,Int}()
    computed_chain = false
    while !computed_chain
      try
        iter = test_solvers_on_single_chain(nsmp, Λ, Ψ, Π, precond, nvec, 
                                            spdim, maxit, verbose=verbose)
        computed_chain = true
      catch err
        nothing
      end
    end

    for (method, _iter) in iter
      if !haskey(iters, method) 
        iters[method] = Array{Int}(undef, nchains)
      end
      iters[method][ichain] = _iter
    end

    println("\n\n ... done working on chain $ichain / $nchains.")

    for (method, _iters) in iters
      npzwrite("data/$(root_fname)_$(method)_nvec$(nvec)_spdim$(spdim).it.npz", _iters[1:ichain])
    end

  end # while ichain <= nchains

  return iters
end


"""
test_solvers_on_several_chains_pll(nchains::Int,
                                   nsmp::Int,
                                   Λ::Array{Float64,1},
                                   Ψ::Array{Float64,2},
                                   Π,
                                   preconds::String,
                                   nvec::Int,
                                   spdim::Int
                                   maxit::Int;
                                   verbose=true)

"""
function test_solvers_on_several_chains_pll(nchains::Int,
                                            nsmp::Int,
                                            Λ::Array{Float64,1},
                                            Ψ::Array{Float64,2},
                                            Π,
                                            precond::String,
                                            nvec::Int,
                                            spdim::Int,
                                            maxit::Int;
                                            verbose=true)

  @everywhere begin
    verbose = $verbose

    iters = Dict{String,Array{Int}}()

    for ichain in 1:nchains

      println("\n\nworking on chain $ichain / $nchains ...")

      iter = test_solvers_on_single_chain(nsmp, Λ, Ψ, Π, precond, nvec, 
                                          spdim, maxit, verbose=verbose)

      for (method, _iter) in iter
        if !haskey(iters, method) 
          iters[method] = Array{Int}(undef, nchains)
        end
        iters[method][ichain] = _iter
      end

      println("\n\n ... done working on chain $ichain / $nchains.")

    end # while ichain <= nchains
  end # @everywhere begin

  # Gather solver iterations from different workers
  for worker in workers()
    iters_of_worker = @getfrom worker iters
    for (method, _iters) in iters
      iters[method] = vcat(_iters, iters_of_worker[method])
    end
  end

  for (method, _iters) in iters
    npzwrite("data/$(root_fname)_$(method)_nvec$(nvec)_spdim$(spdim).it.npz", _iters)
  end

  return iters
end