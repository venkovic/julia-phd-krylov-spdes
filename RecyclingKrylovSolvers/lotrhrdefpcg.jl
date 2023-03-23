"""
     lotrhrdefpcg(A::Union{SparseMatrixCSC{T},
                           FunctionMap},
                  b::Array{T,1},
                  W::Array{T,2},
                  x::Array{T,1},
                  M) 

Performs LO-TR-HR-Deflated-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M 
along with a deflation subspace Span{w1, w2, ...} spanned by the linearly 
independent vectors w1:=W[:,1], w2:=W[:,2], ... Returns basis of new 
deflation subspaces computed by LO-TR-HR projection.

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
function lotrhrdefpcg(A::Union{SparseMatrixCSC{T},
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
  P = Array{T,2}(undef, n, spdim-nvec)
  AP = Array{T,2}(undef, n, spdim-nvec)
  VtAinvMAV = Array{T,2}(undef, spdim, spdim)
  invMAW = Array{T,2}(undef, n, nvec)
  invMAW2 = Array{T,2}(undef, n, nvec)
  invMAP = Array{T,2}(undef, n, spdim-nvec)
  VtAV = Array{T,2}(undef, spdim, spdim)
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
  P[:, it] .= p
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
    P[:, ivec] .= p
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

      for ivec in 1:nev
        invMAW2[:, ivec] = M \ W2tA[ivec, :]
      end
      if isa(A, FunctionMap)
        for ivec in 1:(spdim-nev)
          AP[:, ivec] .= A * P[:, ivec]
        end
      else
        mul!(AP, A, P)
      end
      for ivec in 1:spdim-nev
        invMAP[:, ivec] = M \ AP[:, ivec]
      end
      VtAinvMAV = Array{T,2}(undef, spdim, spdim)
      VtAinvMAV[1:nev, 1:nev] = W2tA * invMAW2
      VtAinvMAV[1:nev, nev+1:spdim] = W2tA * invMAP[:, 1:spdim-nev]
      VtAinvMAV[nev+1:spdim, 1:nev] = VtAinvMAV[1:nev, nev+1:spdim]'
      VtAinvMAV[nev+1:spdim, nev+1:spdim] = AP[:, 1:spdim-nev]'invMAP[:, 1:spdim-nev]
      VtAinvMAV = Symmetric(VtAinvMAV)
    
      VtAV = Array{T,2}(undef, spdim, spdim)      
      VtAV[1:nev, 1:nev] = W2tAW2
      VtAV[1:nev, nev+1:spdim] .= 0
      VtAV[1:nev, nev+1:spdim] .= 0
      VtAV[nev+1:spdim, nev+1:spdim] .= 0
      for ivec in 1:spdim-nev
        #VtAV[nev+ivec, nev+ivec] = P[:, ivec]'AP[:, ivec] # bugs inexplicably
        VtAV[nev+ivec, nev+ivec] = P[:, ivec]'A*P[:, ivec]
      end
      VtAV = Symmetric(VtAV)
    
      Y = zeros(T, spdim, 2 * nvec)
      Y[:, 1:nvec] = eigvecs(VtAinvMAV, VtAV)[:, 1:nvec] # spdim-by-nvec      
      Y[1:spdim-1, nvec+1:end] = eigvecs(VtAinvMAV[1:spdim-1, 1:spdim-1], VtAV[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Array{T,2}(undef, nev, nev)
      H = Q' * (VtAinvMAV * Q) # nev-by-nev
      G = Q' * (VtAV * Q) # nev-by-nev
      vals, vecs = eigen(H, G)
      W2temp .= W2
      W2 = Array{T,2}(undef, n, nev)
      W2tA = Array{T,2}(undef, nev, n)
      W2tAW2 = Array{T,2}(undef, nev, nev)
      invMAW2 = Array{T,2}(undef, n, nev)
      for ivec in 1:nev
        W2[:, ivec] = W2temp * Q[1:size(W2temp)[2], :] * vecs[:, ivec] .+ 
                      P[:, 1:spdim-size(W2temp)[2]] * Q[size(W2temp)[2]+1:spdim, :] * vecs[:, ivec]
      end
      W2temp = Array{T,2}(undef, n, nev)
      ivec = 0

      restarted_once = true
    end
  end

  return x, it, res_norm[1:it], W2[:, 1:nvec]
end


"""
     lotrhrpcg(A::Union{SparseMatrixCSC{T},
                        FunctionMap},
               b::Array{T,1},
               x::Array{T,1},
               M
               nvec::Int,
               spdim::Int) 

Performs LO-TR-RR-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M. 
Returns basis of new deflation subspaces computed by LO-TR-HR projection.

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
function lotrhrpcg(A::Union{SparseMatrixCSC{T},
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
  P = Array{T,2}(undef, n, spdim)
  AP = Array{T,2}(undef, n, spdim)
  VtAinvMAV = Array{T,2}(undef, spdim, spdim)
  invMAP = Array{T,2}(undef, n, spdim)
  invMAW2 = Array{T,2}(undef, n, nvec)
  VtAV = Array{T,2}(undef, spdim, spdim)
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
  P[:, it] = p
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
    P[:, ivec] = p

    if !restarted_once && ivec == spdim

      if isa(A, FunctionMap)
        for ivec in 1:spdim
          AP[:, ivec] .= A * P[:, ivec]
        end
      else
        mul!(AP, A, P)
      end
      for ivec in 1:spdim
        invMAP[:, ivec] = M \ AP[:, ivec]
      end
      VtAinvMAV[1:spdim, 1:spdim] = AP'invMAP
      VtAinvMAV = Symmetric(VtAinvMAV)
        
      VtAV .= 0
      for ivec in 1:spdim
        VtAV[ivec, ivec] = P[:, ivec]'AP[:, ivec]
      end
      VtAV = Symmetric(VtAV)
        
      Y = zeros(T, spdim, 2 * nvec)
      Y[:, 1:nvec] = eigvecs(VtAinvMAV, VtAV)[:, 1:nvec] # spdim-by-nvec      
      Y[1:spdim-1, nvec+1:end] = eigvecs(VtAinvMAV[1:spdim-1, 1:spdim-1], VtAV[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Array{T,2}(undef, nev, nev)
      H = Q' * (VtAinvMAV * Q) # nev-by-nev
      G = Q' * (VtAV * Q) # nev-by-nev
      vals, vecs = eigen(H, G)
      W2 = Array{T,2}(undef, n, nev)
      W2tA = Array{T,2}(undef, nev, n)
      W2tAW2 = Array{T,2}(undef, nev, nev)   
      invMAW2 = Array{T,2}(undef, n, nev)   
      for ivec in 1:nev
        W2[:, ivec] = P * Q * vecs[:, ivec]
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

      for ivec in 1:nev
        invMAW2[:, ivec] = M \ W2tA[ivec, :]
      end
      if isa(A, FunctionMap)
        for ivec in 1:(spdim-nev)
          AP[:, ivec] .= A * P[:, ivec]
        end
      else
        mul!(AP[:, 1:spdim-nev], A, P[:, 1:spdim-nev])
      end
      for ivec in 1:spdim-nev
        invMAP[:, ivec] = M \ AP[:, ivec]
      end
      VtAinvMAV = Array{T,2}(undef, spdim, spdim)
      VtAinvMAV[1:nev, 1:nev] = W2tA * invMAW2
      VtAinvMAV[1:nev, nev+1:spdim] = W2tA * invMAP[:, 1:spdim-nev]
      VtAinvMAV[nev+1:spdim, 1:nev] = VtAinvMAV[1:nev, nev+1:spdim]'
      VtAinvMAV[nev+1:spdim, nev+1:spdim] = AP[:, 1:spdim-nev]'invMAP[:, 1:spdim-nev]
      VtAinvMAV = Symmetric(VtAinvMAV)
    
      VtAV = Array{T,2}(undef, spdim, spdim)
      VtAV[1:nev, 1:nev] = W2tAW2
      VtAV[1:nev, nev+1:spdim] .= 0
      VtAV[1:nev, nev+1:spdim] .= 0
      VtAV[nev+1:spdim, nev+1:spdim] .= 0
      for ivec in 1:spdim-nev
        #VtAV[nev+ivec, nev+ivec] = P[:, ivec]'AP[:, ivec] # bugs inexplicably
        VtAV[nev+ivec, nev+ivec] = P[:, ivec]'A*P[:, ivec]
      end
      VtAV = Symmetric(VtAV)

      Y = zeros(T, spdim, 2 * nvec)
      Y[:, 1:nvec] = eigvecs(VtAinvMAV, VtAV)[:, 1:nvec] # spdim-by-nvec      
      Y[1:spdim-1, nvec+1:end] = eigvecs(VtAinvMAV[1:spdim-1, 1:spdim-1], VtAV[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Array{T,2}(undef, nev, nev)
      H = Q' * (VtAinvMAV * Q) # nev-by-nev
      G = Q' * (VtAV * Q) # nev-by-nev
      vals, vecs = eigen(H, G)
      W2temp .= W2
      W2 = Array{T,2}(undef, n, nev)
      W2tA = Array{T,2}(undef, nev, n)
      W2tAW2 = Array{T,2}(undef, nev, nev)
      invMAW2 = Array{T,2}(undef, n, nev)
      for ivec in 1:nev
        W2[:, ivec] = W2temp * Q[1:size(W2temp)[2], :] * vecs[:, ivec] .+ 
                      P[:, 1:spdim-size(W2temp)[2]] * Q[size(W2temp)[2]+1:spdim, :] * vecs[:, ivec]
      end
      W2temp = Array{T,2}(undef, n, nev)
      ivec = 0
    end

  end

  return x, it, res_norm[1:it], W2[:, 1:nvec]
end