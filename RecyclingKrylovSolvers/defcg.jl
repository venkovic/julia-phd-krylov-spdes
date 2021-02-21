"""
     defcg(A::SparseMatrixCSC{T},
           b::Array{T,1},
           x::Array{T,1},
           W::Array{T,2})

Performs Deflated-CG (Saad et al., 2000).

Used to solve A x = b with an SPD matrix A when a set of linearly independent
vectors w1, w2, ... is known such that Span{w1, w2, ...} is "approximately"
invariant under the action of A. The sequence of iterates of Def-CG is
equivalent to a post-processed sequence of the regular CG solve of a deflated
version of the linear system, with guaranteed decrease of the condition
number. Remark: if Span{w1, w2, ...} is exactly invariant under the action of A,
one should use Init-CG instead of Def-CG because both algorithms would then have
equally positive impacts on convergence, but Def-CG requires an additional
computational cost at every solver iteration.

Saad, Y.; Yeung, M.; Erhel, J. & Guyomarc'h, F.
Deflated Version of the Conjugate Gradient Algorithm,
SIAM Journal on Scientific Computing, SIAM, 1999, 21, 1909-1926.

"""
function defcg(A::Union{SparseMatrixCSC{T},
                        FunctionMap},
               b::Array{T,1},
               x::Array{T,1},
               W::Array{T,2};
               maxit=0)

  n, nvec = size(W)
  r = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)
  mu = Array{T,1}(undef, nvec)

  WtA = W' * A
  WtAW = WtA * W

  r .= b .- A * x

  mu .= W' * r
  mu .= WtAW \ mu
  x .+= W * mu

  maxit == 0 ? maxit = n : nothing
  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  mu .= WtAW \ (WtA * r)
  p .= r .- (W * mu)
  res_norm[it] = sqrt(rTr)

  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < maxit) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTr / d
    beta = 1. / rTr
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    beta *= rTr
    mu .= WtAW \ (WtA * r)
    p .= beta * p .+ r .- (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
  end

  return x, it, res_norm[1:it]
end


"""
     eigdefcg(A::Union{SparseMatrixCSC{T},
                       FunctionMap},
              b::Array{T,1},
              x::Array{T,1},
              W::Array{T,2},
              spdim::Int)

Performs RR-LO-TR-Def-CG (Venkovic et al., 2020), here referred to as eigDef-CG.

Works as a combination of eigCG and Def-CG. The linear solve is deflated as in
Def-CG, and approximate least dominant eigenvectors of A are computed throughout
the solve in a similar way as in eigCG. This algorithm is an alternative to
the incremental eigCG algorithm when solving for a sequence of systems A xs = bs
with a constant SPD matrix A and different right-hand sides bs. This algorithm
should be the method of choice when solving a sequence of linear systems of the
form As xs = bs with correlated SPD matrices A1, A2, ... Examples are shown
below for each type of problem.

Venkovic, N.; Mycek, P; Giraud, L.; Le Maître, O.
Recycling Krylov subspace strategiesfor sequences of sampled stochastic
elliptic equations,
SIAM Journal on Scientific Computing, SIAM, 2020, under review.

"""
function eigdefcg(A::Union{SparseMatrixCSC{T},
                           FunctionMap},
                  b::Array{T,1},
                  x::Array{T,1},
                  W::Array{T,2},
                  spdim::Int;
                  maxit=0)

  n, nvec = size(W)
  r = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  mu = Array{T,1}(undef, nvec)

  WtA = W' * A
  WtAW = WtA * W

  r .= b .- A * x

  mu .= W' * r
  mu .= WtAW \ mu
  x .+= W * mu
  
  nev = nvec
  V = Array{T,2}(undef, n, spdim)
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, spdim, 2 * nvec)
  first_restart = true

  maxit == 0 ? maxit = n : nothing
  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  mu = WtAW \ (WtA * r)
  p .= r .- (W * mu)
  res_norm[it] = sqrt(rTr)

  VtAV[1:nvec, 1:nvec] = WtAW
  V[:, 1:nvec] = W
  
  ivec = nvec + 1
  V[:, ivec] = r / res_norm[it]

  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < maxit) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTr / d
    beta = 1. / rTr
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    beta *= rTr
    mu .= WtAW \ (WtA * r)
    p .= beta * p .+ r .- (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)

    VtAV[ivec, ivec] += 1 / alpha

    if ivec == spdim
      if first_restart
        VtAV[1:nvec, nvec+1:spdim] = WtA * V[:, nvec+1:spdim]
        first_restart = false
      end
      Tm = Symmetric(VtAV) # spdim-by-spdim
      Y[:, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # spdim-by-nvec
      Y[1:spdim-1, nvec+1:end] = eigvecs(Tm[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      vals, Z = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      V[:, 1:nev] = V * (Q * Z) # n-by-nev

      ivec = nev + 1
      V[:, ivec] = r / res_norm[it]
      VtAV .= 0
      for j in 1:nev
        VtAV[j, j] = vals[j]
      end
      VtAV[ivec, ivec] = beta / alpha
    else
      ivec += 1
      V[:, ivec] = r / res_norm[it]
      VtAV[ivec - 1, ivec] = - sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
    end
  end

  return x, it, res_norm[1:it], V[:, 1:nvec]
end


"""
     defpcg(A::Union{SparseMatrixCSC{T},
                     FunctionMap},
            b::Array{T,1},
            W::Array{T,2},
            x::Array{T,1},
            M) 

Performs Deflated-PCG (Saad et al., 2000).

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M, when a
set of linearly independent vectors w1, w2, ... is known such that
Span{w1, w2, ...} is "approximately" invariant under the action of M^{-1}A.
The sequence of iterates of Def-PCG is equivalent to a post-processed sequence
of the regular CG solve of a deflated and split-preconditioned version of the
linear system, with guaranteed decrease of the condition number.
Remark: if Span{w1, w2, ...} is exactly invariant under the action of M^{-1}A,
one should use Init-PCG instead of Def-PCG because both algorithms would then
have equally positive impacts on convergence, but Def-PCG requires an additional
computational cost at every solver iteration.

Saad, Y.; Yeung, M.; Erhel, J. & Guyomarc'h, F.
Deflated Version of the Conjugate Gradient Algorithm,
SIAM Journal on Scientific Computing, SIAM, 1999, 21, 1909-1926.

"""
function defpcg(A::Union{SparseMatrixCSC{T},
                         FunctionMap},
                b::Array{T,1},
                W::Array{T,2},
                x::Array{T,1},
                M;
                maxit=0)

  n, nvec = size(W)
  r = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  z = Array{T,1}(undef, n)
  mu = Array{T,1}(undef, nvec)

  WtA = W' * A
  WtAW = WtA * W

  r .= b .- A * x

  mu .= W' * r
  mu .= WtAW \ mu
  x .+= W * mu

  maxit == 0 ? maxit = n : nothing
  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  res_norm[it] = sqrt(rTr)
  z .= M \ r
  rTz = dot(r, z)
  mu .= WtAW \ (WtA * z)
  p .= z .- (W * mu)
  
  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < maxit) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap .= A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x .+= alpha * p
    axpy!(-alpha, Ap, r) # r .-= alpha * Ap
    rTr = dot(r, r)
    z .= M \ r
    rTz = dot(r, z)
    beta *= rTz
    mu .= WtAW \ (WtA * z)
    p .= beta * p .+ z .- (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
  end

  return x, it, res_norm[1:it]
end


"""
     eigdefpcg(A::Union{SparseMatrixCSC{T},
                        FunctionMap},
               b::Array{T,1},
               x::Array{T,1},
               M,
               W::Array{T,2},
               spdim::Int)

Performs RR-LO-TR-Def-PCG (Venkovic et al., 2020), here referred to as eigDef-PCG.

Works as a combination of eigPCG and Def-PCG. The linear solve is deflated as in
Def-PCG, and approximate least dominant right eigenvectors of M^{-1}A are
computed throughout the solve in a similar way as in eigPCG. This algorithm is
an alternative to the incremental eigPCG algorithm when solving for a sequence
of systems A xs = bs with constant SPD A and M, and different right-hand sides
bs. This algorithm should be the method of choice when solving a sequence of
linear systems of the form As xs = bs with correlated SPD matrices A1, A2, ...
Examples are shown below for each type of problem.

Venkovic, N.; Mycek, P; Giraud, L.; Le Maître, O.
Recycling Krylov subspace strategiesfor sequences of sampled stochastic
elliptic equations,
SIAM Journal on Scientific Computing, SIAM, 2020, under review.

"""
function eigdefpcg(A::Union{SparseMatrixCSC{T},
                            FunctionMap},
                   b::Array{T,1},
                   x::Array{T,1},
                   M,
                   W::Array{T,2},
                   spdim::Int;
                   maxit=0)

  n, nvec = size(W)
  r = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  z = Array{T,1}(undef, n)
  mu = Array{T,1}(undef, nvec)
  
  WtA = W'A
  WtAW = WtA * W
  WtW = W'W

  r .= b .- A * x
  
  mu .= W' * r
  mu .= WtAW \ mu
  x .+= W * mu

  nev = nvec
  V = Array{T,2}(undef, n, spdim)
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, spdim, 2 * nvec)
  first_restart = true

  maxit == 0 ? maxit = n : nothing
  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  z .= M \ r
  rTz = dot(r, z)
  mu .= WtAW \ (WtA * z)
  p .= z .- (W * mu)
  res_norm[it] = sqrt(rTr)

  VtAV[1:nvec, 1:nvec] = WtAW
  V[:, 1:nvec] = W
  just_restarted = false

  ivec = nvec + 1
  V[:, ivec] = z / sqrt(rTz)

  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < maxit) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    r .-= W * (WtW \ (W' * r))
    rTr = dot(r, r)
    z .= M \ r
    rTz = dot(r, z)
    beta *= rTz
    mu .= WtAW \ (WtA * z)
    p .= beta * p .+ z .- (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)

    VtAV[ivec, ivec] += 1 / alpha

    if ivec == spdim
      if first_restart
        VtAV[1:nvec, nvec+1:spdim] = WtA * V[:, nvec+1:spdim]
        first_restart = false
      end
      Tm = Symmetric(VtAV) # spdim-by-spdim
      Y[:, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # spdim-by-nvec
      Y[1:spdim-1, nvec+1:end] = eigvecs(Tm[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      #vals, Z = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      vals, Z = eigen(Symmetric(H))::Eigen{T,T,Array{T,2},Array{T,1}}
      V[:, 1:nev] = V * (Q * Z) # n-by-nev

      ivec = nev + 1
      V[:, ivec] = z / sqrt(rTz)
      VtAV .= 0
      for j in 1:nev
        VtAV[j, j] = vals[j]
      end
      VtAV[ivec, ivec] = beta / alpha
      just_restarted = true
    else
      just_restarted = false
      ivec += 1
      V[:, ivec] = z / sqrt(rTz)
      VtAV[ivec - 1, ivec] = - sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
    end
  end
  if !just_restarted
    if ivec > nvec
      ivec -= 1
      if first_restart
        VtAV[1:nvec, nvec+1:ivec] = WtA * V[:, nvec+1:ivec]
      end
      Tm = Symmetric(VtAV[1:ivec, 1:ivec]) # ivec-by-ivec
      Y .= 0
      Y[1:ivec, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # ivec-by-nvec
      Y[1:ivec-1, nvec+1:end] = eigvecs(Tm[1:ivec-1, 1:ivec-1])[:, 1:nvec] # (ivec-1)-by-nvec
      nev = rank(Y[1:ivec, :]) # nvec <= nev <= 2*nvec
      Q = svd(Y[1:ivec, :]).U[:, 1:nev] # ivec-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      #vals, Z = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      vals, Z = eigen(Symmetric(H))::Eigen{T,T,Array{T,2},Array{T,1}}      
      V[:, 1:nev] = V[:, 1:ivec] * (Q * Z) # n-by-nev
    else
      println("Warning: Less CG iterations than the number of ",
              "eigenvectors wanted. Only Lanczos vectors may be returned.")
    end
  end

  return x, it, res_norm[1:it], V[:, 1:nvec]
end