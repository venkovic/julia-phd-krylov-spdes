using SparseArrays: SparseMatrixCSC
using LinearMaps: FunctionMap

"""
     get_constant_preconds()

Returns queried constant preconditioners.

"""
function get_constant_preconds(preconds::Array{String,1},
                               A_0::SparseMatrixCSC{Float64,Int})

  Π, Π_IId = [], []

  do_dd = false
  for precond in preconds
    if precond in preconds_with_dd
      do_dd = true
      break
    end
  end

  for precond in preconds

    if precond == "amg_0"
      printlnln("prepare amg_0 preconditioner for A_0 ...")
      Π_amg_0 = @time AMGPreconditioner{SmoothedAggregation}(A_0);
      push!(Π, Π_amg_0)
      flush(stdout)

    elseif precond == "lorasc$(ndom)_0"
      printlnln("prepare_lorasc_precond for A_0 with ε = 0 ...")
      Π_lorasc_0 = @time prepare_lorasc_precond(ndom,
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
      push!(Π, Π_lorasc_0)
      flush(stdout)

    elseif precond == "lorasc$(ndom)_1"
      printlnln("prepare_lorasc_precond for A_0 with ε = 0.01 ...")
      Π_lorasc_1 = @time prepare_lorasc_precond(ndom,
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
      push!(Π, Π_lorasc_1)
      flush(stdout)

    elseif precond == "bj$(nbj)_0"
      printlnln("prepare bj_0 preconditioner for A_0 ...")
      Π_bj_0 = @time BJPreconditioner(nbj, A_0);
      push!(Π, Π_bj_0)
      flush(stdout)

    elseif precond == "chol16"
      printlnln("prepare chol16_0 preconditioner for A_0 ...")
      Π_chol16_0 = @time get_cholesky16(A_0);
      push!(Π, Π_chol16_0)
      flush(stdout)

    elseif precond == "neumann-neumann$(ndom)_0"
      printlnln("prepare neumann-neummann preconditioner for A_0 ...")
      ΠS_nn_0, _Π_IId = prepare_neumann_neumann_schur_precond(ndom,
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
                                                             f,
                                                             uexact,
                                                             load_partition=false)
      push!(Π, ΠS_nn_0)
      push!(Π_IId, _Π_IId)
      flush(stdout)

    end

  end # precond in preconds

  if do_dd
    return Π, Π_IId[1]
  else
    return Π
  end

end # function get_constant_preconds


"""
test_solvers_on_a_chain(nsmp::Int,
                        Λ::Array{Float64,1},
                        Ψ::Array{Float64,2}
                        Π,
                        precond_labels::Array{String,1},
                        do_pcg::Bool,
                        do_eigpcg::Bool,
                        do_eigdefpcg::Bool,
                        do_defpcg::Bool,
                        nvec::Int,
                        spdim::Int;
                        verbose=true,
                        save_results=false)


"""
function test_solvers_on_single_chain(nsmp::Int,
                                      Λ::Array{Float64,1},
                                      Ψ::Array{Float64,2},
                                      Π,
                                      preconds::Array{String,1},
                                      nvec::Int,
                                      spdim::Int,
                                      maxit::Int;
                                      do_pcg=true,
                                      do_eigpcg=true,
                                      do_eigdefpcg=true,
                                      do_defpcg=true,
                                      verbose=true,
                                      save_results=false,
                                      troubleshoot=false,
                                      Π_IId=nothing)

  status = 0 

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

  do_schur = false
  for precond in preconds
    occursin("neumann-neumann", precond) ? do_schur = true : nothing
  end

  if do_schur
    S, b_schur = do_condensed_isotropic_elliptic_assembly(ndom,
                                                          cells,
                                                          points,
                                                          epart,
                                                          ind_Id_g2l,
                                                          ind_Γd_g2l,
                                                          ind_Γ_g2l,
                                                          ind_Γd_Γ2l,
                                                          node_owner,
                                                          node_Γ_cnt,
                                                          exp.(sampler.g),
                                                          f,
                                                          uexact,
                                                          Π_IId)
    n_Γ, = size(b_schur)
    x_schur = zeros(Float64, n_Γ)
    WtS = Array{Float64,2}(undef, nvec, n_Γ)
  end

  methods = String[]
  W = Dict{String,Array{Float64,2}}()
  WtA = Array{Float64,2}(undef, nvec, A.n)
  iter = Dict{String,Array{Int,1}}()

  for precond in preconds
    if do_pcg
      method = precond * "-pcg"
      push!(methods, method)
      iter[method] = Array{Int,1}(undef, nsmp)
    end

    if do_eigpcg 
      method = precond * "-eigpcg"
      push!(methods, method)
      iter[method] = Array{Int,1}(undef, nsmp)
      W[method] = Array{Float64,2}(undef, A.n, nvec)
    end

    if do_eigdefpcg
      method = precond * "-eigdefpcg"
      push!(methods, method)
      iter[method] = Array{Int,1}(undef, nsmp)
      W[method] = Array{Float64,2}(undef, A.n, nvec)
    end

    if do_defpcg
      method = precond * "-defpcg"
      push!(methods, method)
      iter[method] = Array{Int,1}(undef, nsmp)
      W[method] = Array{Float64,2}(undef, A.n, nvec)
    end
  end

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

      if do_schur
        S, b_schur = do_condensed_isotropic_elliptic_assembly(ndom,
                                                              cells,
                                                              points,
                                                              epart,
                                                              ind_Id_g2l,
                                                              ind_Γd_g2l,
                                                              ind_Γ_g2l,
                                                              ind_Γd_Γ2l,
                                                              node_owner,
                                                              node_Γ_cnt,
                                                              exp.(sampler.g),
                                                              f,
                                                              uexact,
                                                              Π_IId)
      end
    end # if s > 1

    for (p, precond) in enumerate(preconds)

      if do_pcg
        method = precond * "-pcg"
        verbose ? print("$method of A * u = b ... ") : nothing
        troubleshoot ? save_system(A, b) : nothing 
        try
          if occursin("neumann-neumann", precond)
            x_schur .= 0
            Δt = @elapsed _, it, _ = pcg(S, b_schur, x_schur, Π[p], maxit=maxit)
          else
            x .= 0
            Δt = @elapsed _, it, _  = pcg(A, b, x, Π[p], maxit=maxit)
          end
        catch err
          status = -1
          return iter, status
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
        flush(stdout)
      end

      if do_eigpcg
        method = precond * "-eigpcg"
        verbose ? print("$method of A * u = b ... ") : nothing
        troubleshoot ? save_system(A, b) : nothing 
        try
          if occursin("neumann-neumann", precond)
            if s == 1
              x_schur .= 0
            else
              if isa(S, FunctionMap)
                for j in 1:nvec
                  WtS[j, :] .= S * W[method][:, j] 
                end
              else
                WtS .= W[method]'S
              end
              H = WtS * W[method]
              x_schur .= W[method] * (H \ (W[method]'b_schur))
            end
            Δt = @elapsed _, it, _, W[method] = eigpcg(S, b_schur, x_schur, Π[p], nvec, spdim, maxit=maxit)
          else
            if s == 1
              x .= 0
            else
              WtA .= W[method]'A
              H = WtA * W[method]
              x .= W[method] * (H \ (W[method]'b))
            end
            Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim, maxit=maxit)
          end
        catch err
          throw(err)
          status = -1
          return iter, status
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
        flush(stdout)
      end # if do_eigpcg

      if do_eigdefpcg
        method = precond * "-eigdefpcg"
        verbose ? print("$method of A * u = b ... ") : nothing
        if s == 1
          troubleshoot ? save_system(A, b) : nothing

          #Δt = @elapsed _, it, _, W[method] = eigpcg(S, b_schur, x, Π[p], nvec, spdim, maxit=maxit)
          try
            if occursin("neumann-neumann", precond)
              x_schur .= 0
              Δt = @elapsed _, it, _, W[method] = eigpcg(S, b_schur, x_schur, Π[p], nvec, spdim, maxit=maxit)
            else
              x .= 0
              Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim, maxit=maxit)
            end
          catch err
            status = -1
            println(err)
            return iter, status
          end
        else
          troubleshoot ? save_deflated_system(A, b, W[method]) : nothing
          try
            if rank(W[method]) < .9 * nvec
              status = -1
              return iter, status
            end
            if occursin("neumann-neumann", precond)
              x_schur .= 0
              Δt = @elapsed _, it, _, W[method] = eigdefpcg(S, b_schur, x_schur, Π[p], W[method], spdim, maxit=maxit)
            else
              x .= 0
              Δt = @elapsed _, it, _, W[method] = eigdefpcg(A, b, x, Π[p], W[method], spdim, maxit=maxit)
            end
          catch err
            status = -1
            return iter, status
          end
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
        flush(stdout)
      end # if do_eigdefpcg

      if do_defpcg
        method = precond * "-defpcg"
        x .= 0
        verbose ? print("$method of A * u = b ... ") : nothing
        if s == 1
          # Some work remains to do here
          troubleshoot ? save_system(A, b) : nothing 
          try
            if occursin("neumann-neumann", precond)
              x_schur .= 0
              Δt = @elapsed _, it, _, W[method] = eigpcg(S, b_schur, x_schur, Π[p], nvec, spdim, maxit=maxit)
            else
              x .= 0
              Δt = @elapsed _, it, _, W[method] = eigpcg(A, b, x, Π[p], nvec, spdim, maxit=maxit)
            end
          catch err
            status = -1
            return iter, status
          end

        else
          troubleshoot ? save_deflated_system(A, b, W[method]) : nothing 
          try
            if occursin("neumann-neumann", precond)
              x_schur .= 0
              Δt = @elapsed _, it, _, W[method] = defpcg(S, b_schur, W[method], x_schur, Π[p], maxit=maxit)
            else
              x .= 0
              Δt = @elapsed _, it, _, W[method] = defpcg(A, b, W[method], x, Π[p], maxit=maxit)
            end            
          catch err
            status = -1
            return iter, status
          end
        end
        verbose ? println("$Δt seconds, iter = $it") : nothing
        iter[method][s] = it
        flush(stdout)
      end # if do_defpcg

    end # for (p, precond)
  end # for s in 1:nsmp

  if save_results
    for method in methods
      npzwrite("data/$(root_fname)_$method.it.npz", iter[method])
    end
  end

  return iter, status
end


"""
test_solvers_on_several_chains(nchains::Int,
                               nsmp::Int,
                               Λ::Array{Float64,1},
                               Ψ::Array{Float64,2},
                               Π,
                               preconds::Array{String,1},
                               do_pcg::Bool,
                               do_eigpcg::Bool,
                               do_eigdefpcg::Bool,
                               do_defpcg::Bool,
                               nvec::Int,
                               spdim::Int;
                               verbose=true,
                               save_results=true)

"""
function test_solvers_on_several_chains(nchains::Int,
                                        nsmp::Int,
                                        Λ::Array{Float64,1},
                                        Ψ::Array{Float64,2},
                                        Π,
                                        preconds::Array{String,1},
                                        nvec::Int,
                                        spdim::Int,
                                        maxit::Int;
                                        do_pcg=true,
                                        do_eigpcg=true,
                                        do_eigdefpcg=true,
                                        do_defpcg=true,
                                        verbose=true,
                                        save_results=true,
                                        troubleshoot=false,
                                        Π_IId=nothing)

  iters = Dict{String,Array{Int,2}}()

  ichain = 1
  while ichain <= nchains

    println("\n\nworking on chain $ichain / $nchains ...")

    iter, status = test_solvers_on_single_chain(nsmp, Λ, Ψ, Π, preconds, nvec, spdim, maxit,
                                                do_pcg=do_pcg,
                                                do_eigpcg=do_eigpcg,
                                                do_eigdefpcg=do_eigdefpcg,
                                                do_defpcg=do_defpcg,
                                                verbose=verbose,
                                                troubleshoot=troubleshoot,
                                                Π_IId=Π_IId)

    if status == 0
      for (method, _iter) in iter
        if haskey(iters, method) 
          iters[method] = hcat(iters[method], _iter)
        else
          iters[method] = reshape(_iter, length(_iter), 1)
        end
      end

      if save_results
        for (method, _iters) in iters
          if occursin("-eigpcg", method) | occursin("-eigdefpcg", method) | occursin("-defpcg", method)
            npzwrite("data/$(root_fname)_$(method)_nvec$(nvec)_spdim$(spdim).it.npz", _iters)
          else
            npzwrite("data/$(root_fname)_$(method).it.npz", _iters)
          end
        end
      end

      println("\n\n ... done working on chain $ichain / $nchains.")
      ichain += 1

    end # if status = 0
  end # while ichain <= nchains

  return iters
end