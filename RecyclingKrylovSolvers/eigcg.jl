"""
     eigcg(A::Union{SparseMatrixCSC{T},
                    FunctionMap},
           b::Array{T,1},
           x::Array{T,1},
           nvec::Int,
           spdim::Int)

Performs eigCG (Stathopoulos & Orginos, 2010).

Used at the beginning of a solving procedure of linear systems A xs = bs with a
constant SPD matrix A, and different right-hand sides bs. eigCG may be run once
(or incrementally) to generate approximate least dominant eigenvectors of A.
These approximate eigenvectors are then used to generate a deflated initial
guess with the Init-CG algorithm. Incremental eigCG should be used when the
solve of the first system ends before accurate eigenvector approximations can be
obtained by eigCG, which then limits the potential speed-up obtained for the
subsequent Init-CG solves. See Example for typical use and implementation of the
Incremental eigCG algorithm (Stathopoulos & Orginos, 2010).

Stathopoulos, A. & Orginos, K.
Computing and deflating eigenvalues while solving multiple right-hand side
linear systems with an application to quantum chromodynamics,
SIAM Journal on Scientific Computing, SIAM, 2010, 32, 439-462.

"""
function eigcg(A::Union{SparseMatrixCSC{T},
                        FunctionMap},
               b::Array{T,1},
               x::Array{T,1},
               nvec::Int,
               spdim::Int)

  n, = size(x)
  r = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)

  V = Array{T,2}(undef, n, spdim)
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, spdim, 2 * nvec)
  tvec = Array{T,1}(undef, n)
  just_restarted = false

  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  p .= r
  res_norm[it] = sqrt(rTr)
  ivec = 1
  V[:, ivec] = r / res_norm[it]

  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTr / d
    beta = 1. / rTr
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    beta *= rTr
    if ivec == spdim
      tvec .-= beta * Ap
    end
    axpby!(1, r, beta, p) # p = beta * p + r
    it += 1
    res_norm[it] = sqrt(rTr)

    VtAV[ivec, ivec] += 1 / alpha
    if just_restarted
      tvec .+= Ap
      nev = ivec - 1
      VtAV[1:nev, ivec] = V[:, 1:nev]' * (tvec / res_norm[it - 1])
      just_restarted = false
    end

    if ivec == spdim
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
      tvec .= - beta * Ap
      just_restarted = true
      #VtAV[1:nev, ivec] = V[:, 1:nev]' * (A * V[:, ivec]) # Matrix-vector product avoided with tvec
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
     eigpcg(A::SparseMatrixCSC{T}, 
            b::Array{T,1},
            x::Array{T,1},
            M,
            nvec::Int,
            spdim::Int)

Performs eigPCG (Stathopoulos & Orginos, 2010).

Used at the beginning of a solving procedure of linear systems A xs = bs with
constant SPD matrix A and SPD preconditioner M, and different right-hand sides
bs. eigPCG may be run once (or incrementally) to generate approximate least
dominant right eigenvectors of M^{-1}A. These approximate eigenvectors are then
used to generate a deflated initial guess with the Init-PCG algorithm.
Incremental eigPCG should be used when the solve of the first system ends before
accurate eigenvector approximations can be obtained by eigPCG, which then limits
the potential speed-up obtained for the subsequent Init-PCG solves. See Examples
for typical use and implementation of the Incremental eigPCG algorithm
(Stathopoulos & Orginos, 2010).

Stathopoulos, A. & Orginos, K.
Computing and deflating eigenvalues while solving multiple right-hand side
linear systems with an application to quantum chromodynamics,
SIAM Journal on Scientific Computing, SIAM, 2010, 32, 439-462.

"""
function eigpcg(A::SparseMatrixCSC{T}, 
                b::Array{T,1},
                x::Array{T,1},
                M,
                nvec::Int,
                spdim::Int)
                
  n, = size(x)
  r = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)

  V = Array{T,2}(undef, n, spdim)
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, spdim, 2 * nvec)
  tvec = Array{T,1}(undef, n)
  just_restarted = false

  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  z .= M \ r
  rTz = dot(r, z)
  p .= z
  res_norm[it] = sqrt(rTr)

  ivec = 1
  V[:, ivec] = z / sqrt(rTz)
  just_restarted = false

  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    z .= M \ r
    if just_restarted
      hlpr = sqrt(rTz)
    end
    rTz = dot(r, z)
    beta *= rTz
    if ivec == spdim
      tvec .-= beta * Ap
    end
    axpby!(1, z, beta, p) # p = beta * p + z
    it += 1
    res_norm[it] = sqrt(rTr)
    #
    VtAV[ivec, ivec] += 1 / alpha
    if just_restarted
      tvec .+= Ap
      nev = ivec - 1
      VtAV[1:nev, ivec] = V[:, 1:nev]' * (tvec / hlpr)
      just_restarted = false
    end
    #
    if ivec == spdim
      VtAV = V' * (A * V)
      Tm = Symmetric(VtAV) # spdim-by-spdim
      Y[:, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # spdim-by-nvec
      Y[1:spdim-1, nvec+1:end] = eigvecs(Tm[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      #println("H:", typeof(H), ", ", issymmetric(H))
      vals, Z = eigen(Symmetric(H))::Eigen{T,T,Array{T,2},Array{T,1}}
      V[:, 1:nev] = V * (Q * Z) # n-by-nev
      #
      ivec = nev + 1
      V[:, ivec] = z / sqrt(rTz)
      VtAV .= 0
      for j in 1:nev
        VtAV[j, j] = vals[j]
      end
      VtAV[ivec, ivec] = beta / alpha
      tvec .= - beta * Ap
      just_restarted = true
      #VtAV[1:nev, ivec] = V[:, 1:nev]' * (A * V[:, ivec]) # Matrix-vector product avoided with tvec
    else
      ivec += 1
      V[:, ivec] = z / sqrt(rTz)
      VtAV[ivec - 1, ivec] = - sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
    end
  end
  if !just_restarted
    if ivec > nvec
      ivec -= 1
      Tm = Symmetric(VtAV[1:ivec, 1:ivec]) # ivec-by-ivec
      Y .= 0
      Y[1:ivec, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # ivec-by-nvec
      Y[1:ivec-1, nvec+1:end] = eigvecs(Tm[1:ivec-1, 1:ivec-1])[:, 1:nvec] # (ivec-1)-by-nvec
      nev = rank(Y[1:ivec, :]) # nvec <= nev <= 2*nvec
      Q = svd(Y[1:ivec, :]).U[:, 1:nev] # ivec-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      vals, Z = eigen(Symmetric(H))::Eigen{T,T,Array{T,2},Array{T,1}}
      V[:, 1:nev] = V[:, 1:ivec] * (Q * Z) # n-by-nev
    else
      println("Warning: Less CG iterations than the number of ",
              "eigenvectors wanted. Only Lanczos vectors may be returned.")
    end
  end

  return x, it, res_norm[1:it], V[:, 1:nvec]
end
