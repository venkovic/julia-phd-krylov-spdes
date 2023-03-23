"""
     lotrrrdefpcg(A::Union{SparseMatrixCSC{T},
                           FunctionMap},
                  b::Array{T,1},
                  W::Array{T,2},
                  x::Array{T,1},
                  M) 

Performs LO-TR-RR-Deflated-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M 
along with a deflation subspace Span{w1, w2, ...} spanned by the linearly 
independent vectors w1:=W[:,1], w2:=W[:,2], ... Returns basis of new 
deflation subspaces computed by LO-TR-RR projection.

Input:

 `A::Union{SparseMatrixCSC{T},FunctionMap}`,
  matrix of linear system.

 `b::Array{T,1}`,
  right-hand side of linear system.
 
 `W::Array{T,2}`,
  basis vectors of deflation subspace.

 `x::Array{T,1}`,
  initial iterate.
  
 `M`,
  preconditioner.

Output:

 `x::Array{T,1}`,
  solution of linear system.

 `it::Int`,
  number of solver iterations.

 `res_norm::Array{T,1}`,
  residual norm of iterates.

 `W2::Array{T,2}`,
  basis vectors of new deflation subspace.
  
"""
function lotrrrdefpcg(A::Union{SparseMatrixCSC{T},
                               FunctionMap},
                      b::Array{T,1},
                      x::Array{T,1},
                      W::Array{T,2},
                      M,
                      spdim::Int;
                      maxit=0)

  n, nvec = size(W)
  r = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  z = Array{T,1}(undef, n)
  mu = Array{T,1}(undef, nvec)
  WtA = Array{T,2}(undef, nvec, n)
  WtAW = Array{T,2}(undef, nvec, nvec)
  WtW = Array{T,2}(undef, nvec, nvec)
  Z = Array{T,2}(undef, n, spdim-nvec)
  AZ = Array{T,2}(undef, n, spdim-nvec)
  VtAV = Array{T,2}(undef, spdim, spdim)
  VtMV = Array{T,2}(undef, spdim, spdim)
  W2 = Array{T,2}(undef, n, nvec)
  W2temp = Array{T,2}(undef, n, nvec)
  W2tA = Array{T,2}(undef, nvec, n)
  W2tAW2 = Array{T,2}(undef, nvec, nvec)
  
  restarted_once = false

  if isa(A, FunctionMap)
    for ivec in 1:nvec
      WtA[ivec, :] .= A * W[:, ivec]
    end
  else
    mul!(WtA, W', A)
  end
  WtAW = WtA * W
  WtAW = Symmetric(WtAW)
  mul!(WtW, W', W)
  WtW = Symmetric(WtW)
  W2 .= W
  W2tA .= WtA
  W2tAW2 .= WtAW

  # Compute initial iterate with residual orthogonal to Range(W)
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
  Z[:, it] = z ./ rTz^.5
  ivec = 1
  nev = nvec
  
  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < maxit) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap .= A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x .+= alpha * p
    axpy!(-alpha, Ap, r) # r .-= alpha * Ap
    r .-= W * (WtW \ (W'r))
    rTr = dot(r, r)
    z .= M \ r
    rTz = dot(r, z)
    beta *= rTz
    mu .= WtAW \ (WtA * z)
    p .= beta * p .+ z .- (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
    ivec += 1
    Z[:, ivec] = z ./ rTz^.5
    if ivec == spdim - nev      
      if restarted_once
        if isa(A, FunctionMap)
          for ivec in 1:nev
            W2tA[ivec, :] .= A * W2[:, ivec]
          end
        else
          mul!(W2tA, W2', A)
        end
        W2tAW2 = W2tA * W2
        W2tAW2 = Symmetric(W2tAW2)
      end

      VtAV = Array{T,2}(undef, spdim, spdim)
      VtAV[1:nev, 1:nev] = W2tAW2
      VtAV[1:nev, nev+1:spdim] = W2tA * Z[:, 1:spdim-nev]
      VtAV[nvec+1:spdim, 1:nvec] = VtAV[1:nvec, nvec+1:spdim]'
      if isa(A, FunctionMap)
        for ivec in 1:(spdim-nev)
          AZ[:, ivec] .= A * Z[:, ivec]
        end
      else
        mul!(AZ, A, Z)
      end
      VtAV[nev+1:spdim, nev+1:spdim] = Z[:, 1:spdim-nev]'AZ[:, 1:spdim-nev]
      VtAV = Symmetric(VtAV)
      
      VtMV = Array{T,2}(undef, spdim, spdim)
      VtMV[1:nev, 1:nev] = I(nev)
      VtMV[1:nev, nev+1:spdim] .= 0
      VtMV[1:nev, nev+1:spdim] .= 0
      VtMV[nev+1:spdim, nev+1:spdim] = I(spdim-nev)
      VtMV = Symmetric(VtMV)
   
      Y = zeros(T, spdim, 2 * nvec)
      Y[:, 1:nvec] = eigvecs(VtAV, VtMV)[:, 1:nvec] # spdim-by-nvec      
      Y[1:spdim-1, nvec+1:end] = eigvecs(VtAV[1:spdim-1, 1:spdim-1], VtMV[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Array{T,2}(undef, nev, nev)
      H = Q' * (VtAV * Q) # nev-by-nev
      vals, vecs = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      W2temp .= W2
      W2 = Array{T,2}(undef, n, nev)
      W2tA = Array{T,2}(undef, nev, n)
      W2tAW2 = Array{T,2}(undef, nev, nev)
      for ivec in 1:nev
        W2[:, ivec] = W2temp * Q[1:size(W2temp)[2], :] * vecs[:, ivec] .+ 
                      Z[:, 1:spdim-size(W2temp)[2]] * Q[size(W2temp)[2]+1:spdim, :] * vecs[:, ivec]
      end
      W2temp = Array{T,2}(undef, n, nev)
      ivec = 0

      restarted_once = true
    end
  end

  return x, it, res_norm[1:it], W2[:, 1:nvec]
end


"""
     lotrrrpcg(A::Union{SparseMatrixCSC{T},
                        FunctionMap},
               b::Array{T,1},
               x::Array{T,1},
               M
               nvec::Int,
               spdim::Int) 

Performs LO-TR-RR-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M. 
Returns basis of new deflation subspaces computed by LO-TR-RR projection.

Input:

 `A::Union{SparseMatrixCSC{T},FunctionMap}`,
  matrix of linear system.

 `b::Array{T,1}`,
  right-hand side of linear system.

 `x::Array{T,1}`,
  initial iterate.
  
 `M`,
  preconditioner.

 `nvec::Int`,
  number of wanted eigenvectors.

 `spdim::Int`,
  size of the eigen search space.


Output:

 `x::Array{T,1}`,
  solution of linear system.

 `it::Int`,
  number of solver iterations.

 `res_norm::Array{T,1}`,
  residual norm of iterates.

 `W2::Array{T,2}`,
  basis vectors of new deflation subspace.
  
"""
function lotrrrpcg(A::Union{SparseMatrixCSC{T},
                            FunctionMap},
                   b::Array{T,1},
                   x::Array{T,1},
                   M,
                   nvec::Int,
                   spdim::Int;
                   maxit=0)

  n = size(b)[1]
  r = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  z = Array{T,1}(undef, n)
  Z = Array{T,2}(undef, n, spdim)
  AZ = Array{T,2}(undef, n, spdim)
  VtAV = Array{T,2}(undef, spdim, spdim)
  VtMV = Array{T,2}(undef, spdim, spdim)
  W2 = Array{T,2}(undef, n, nvec)
  W2temp = Array{T,2}(undef, n, nvec)
  W2tA = Array{T,2}(undef, nvec, n)
  W2tAW2 = Array{T,2}(undef, nvec, nvec)

  restarted_once = false

  maxit == 0 ? maxit = n : nothing
  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  res_norm[it] = sqrt(rTr)
  z .= M \ r
  rTz = dot(r, z)
  p .= z
  Z[:, it] = z ./ rTz^.5
  ivec = 1
  nev = nvec

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
    p .= beta * p .+ z
    it += 1
    res_norm[it] = sqrt(rTr)
    ivec += 1
    Z[:, ivec] = z ./ rTz^.5

    if !restarted_once && ivec == spdim
      if isa(A, FunctionMap)
        for ivec in 1:spdim
          AZ[:, ivec] .= A * Z[:, ivec]
        end
      else
        mul!(AZ, A, Z)
      end
      VtAV[1:spdim, 1:spdim] = Z'AZ
      VtAV = Symmetric(VtAV)
        
      VtMV[1:spdim, 1:spdim] = I(spdim)
      VtMV = Symmetric(VtMV)

      Y = zeros(T, spdim, 2 * nvec)
      Y[:, 1:nvec] = eigvecs(VtAV, VtMV)[:, 1:nvec] # spdim-by-nvec      
      Y[1:spdim-1, nvec+1:end] = eigvecs(VtAV[1:spdim-1, 1:spdim-1], VtMV[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Array{T,2}(undef, nev, nev)
      H = Q' * (VtAV * Q) # nev-by-nev
      vals, vecs = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      W2 = Array{T,2}(undef, n, nev)
      W2tA = Array{T,2}(undef, nev, n)
      W2tAW2 = Array{T,2}(undef, nev, nev)      
      for ivec in 1:nev
        W2[:, ivec] = Z * Q * vecs[:, ivec]
      end
      W2temp = Array{T,2}(undef, n, nev)
      ivec = 0

      restarted_once = true

    elseif restarted_once && ivec == spdim - nev
      
      if isa(A, FunctionMap)
        for ivec in 1:nev
          W2tA[ivec, :] .= A * W2[:, ivec]
        end
      else
        mul!(W2tA, W2', A)
      end
      W2tAW2 = W2tA * W2
      W2tAW2 = Symmetric(W2tAW2)

      VtAV = Array{T,2}(undef, spdim, spdim)
      VtAV[1:nev, 1:nev] .= W2tAW2
      VtAV[1:nev, nev+1:spdim] = W2tA * Z[:, 1:spdim-nev]
      VtAV[nev+1:spdim, 1:nev] = VtAV[1:nev, nev+1:spdim]'
      if isa(A, FunctionMap)
        for ivec in 1:(spdim-nev)
          AZ[:, ivec] .= A * Z[:, ivec]
        end
      else
        mul!(AZ[:, 1:spdim-nev], A, Z[:, 1:spdim-nev])
      end
      VtAV[nev+1:spdim, nev+1:spdim] = Z[:, 1:spdim-nev]'AZ[:, 1:spdim-nev]
      VtAV = Symmetric(VtAV)
 
      VtMV = Array{T,2}(undef, spdim, spdim)
      VtMV[1:nev, 1:nev] = I(nev)
      VtMV[1:nev, nev+1:spdim] .= 0
      VtMV[1:nev, nev+1:spdim] .= 0
      VtMV[nev+1:spdim, nev+1:spdim] = I(spdim-nev)
      VtMV = Symmetric(VtMV)
   
      Y = zeros(T, spdim, 2 * nvec)
      Y[:, 1:nvec] = eigvecs(VtAV, VtMV)[:, 1:nvec] # spdim-by-nvec      
      Y[1:spdim-1, nvec+1:end] = eigvecs(VtAV[1:spdim-1, 1:spdim-1], VtMV[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Array{T,2}(undef, nev, nev)
      H = Q' * (VtAV * Q) # nev-by-nev
      vals, vecs = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      W2temp .= W2
      W2 = Array{T,2}(undef, n, nev)
      W2tA = Array{T,2}(undef, nev, n)
      W2tAW2 = Array{T,2}(undef, nev, nev)
      for ivec in 1:nev
        W2[:, ivec] = W2temp * Q[1:size(W2temp)[2], :] * vecs[:, ivec] .+ 
                      Z[:, 1:spdim-size(W2temp)[2]] * Q[size(W2temp)[2]+1:spdim, :] * vecs[:, ivec]
      end
      W2temp = Array{T,2}(undef, n, nev)
      ivec = 0
    end    
  end

  return x, it, res_norm[1:it], W2[:, 1:nvec]
end