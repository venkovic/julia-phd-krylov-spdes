using SparseArrays: SparseMatrixCSC
using LinearMaps: FunctionMap

"""
test_solvers_on_multiple_rhs(nsmp::Int,
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
function test_solvers_on_multiple_rhs(nsmp::Int,
                                      Λ::Array{Float64,1},
                                      Ψ::Array{Float64,2},
                                      Π,
                                      preconds::Array{String,1},
                                      nvec::Int,
                                      spdim::Int,
                                      maxit::Int;
                                      do_pcg=true,
                                      do_initpcg=true,
                                      do_eigpcg=true,
                                      do_eigdefpcg=true,
                                      do_defpcg=true,
                                      verbose=true,
                                      save_results=false,
                                      troubleshoot=false,
                                      Π_IId=nothing)

  status = 0 

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

  do_schur = false
  for precond in preconds
    occursin("neumann-neumann", precond) ? do_schur = true : nothing
  end

  if do_schur


    # where are epart, ... ? Are these global :/ ?
     
    verbose ? printlnln("prepare_local_schurs ...") : nothing
    Δt = @elapsed A_IIdd, A_IΓdd, A_ΓΓdd, b_Idd, b_Γ = prepare_local_schurs(cells,
                                                                            points,
                                                                            epart,
                                                                            ind_Id_g2l,
                                                                            ind_Γd_g2l,
                                                                            ind_Γ_g2l,
                                                                            node_owner,
                                                                            a_vec,
                                                                            f,
                                                                            uexact)
    verbose ? println("$Δt seconds") : nothing

    S, b_schur = do_condensed_isotropic_elliptic_assembly(ndom,
                                                          A_IIdd,
                                                          A_IΓdd,
                                                          A_ΓΓdd,
                                                          b_Idd,
                                                          b_Γ,
                                                          ind_Γ_g2l,
                                                          ind_Γd_Γ2l,
                                                          node_Γ_cnt,
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

    if do_initpcg
        method = precond * "-initpcg"
        push!(methods, method)
        iter[method] = Array{Int,1}(undef, nsmp)
        W[method] = Array{Float64,2}(undef, A.n, nvec)
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
  # Sample rhs uniformly at random and solve linear systems with 
  # online eigenvectors approximation
  #
  it = 0
  for s in 1:nsmp

    if s > 1
      verbose ? print("\n$s / $nsmp") : nothing

      b = rand(A.n)

      if do_schur

        domain_decompose_rhs!(b, b_Idd, b_Γ, 
                              ind_Id_g2l,
                              ind_Γ_g2l,
                              node_owner)

        b_schur = get_schur_rhs(b_Idd, A_IIdd, A_IΓdd, b_Γ, ind_Γd_Γ2l, preconds=Π_IId)
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










      if do_initpcg
        method = precond * "-initpcg"
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
      end # if do_initpcg













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
      npzwrite("data/$(root_fname)_$(method)_mrhs_.it.npz", iter[method])
    end
  end

  return iter, status
end