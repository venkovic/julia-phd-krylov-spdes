"""
     trhrdefpcg(A::Union{SparseMatrixCSC{T},
                         FunctionMap},
                b::Array{T,1},
                W::Array{T,2},
                x::Array{T,1},
                M) 

Performs TR-HR-Deflated-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M 
along with a deflation subspace Span{w1, w2, ...} spanned by the linearly 
independent vectors w1:=W[:,1], w2:=W[:,2], ... Returns basis of new 
deflation subspaces computed by TR-HR projection.

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
function trhrdefpcg(A::Union{SparseMatrixCSC{T},
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
    ivec = it % (spdim - nvec)
    ivec == 0 ? ivec = spdim - nvec : nothing
    P[:, ivec] .= p
    if ivec == spdim - nvec
      if restarted_once
        if isa(A, FunctionMap)
          for ivec in 1:nvec
            W2tA[ivec, :] .= A * W2[:, ivec]
          end
        else
          mul!(W2tA, W2', A)
        end
        W2tAW2 = W2tA * W2
        W2tAW2 = Symmetric(W2tAW2)
      end

      for ivec in 1:nvec
        invMAW2[:, ivec] = M \ W2tA[ivec, :]
      end
      if isa(A, FunctionMap)
        for ivec in 1:(spdim-nvec)
          AP[:, ivec] .= A * P[:, ivec]
        end
      else
        mul!(AP, A, P)
      end
      for ivec in 1:spdim-nvec
        invMAP[:, ivec] = M \ AP[:, ivec]
      end
      VtAinvMAV = Array{T,2}(undef, spdim, spdim)
      VtAinvMAV[1:nvec, 1:nvec] = W2tA * invMAW2
      VtAinvMAV[1:nvec, nvec+1:spdim] = W2tA * invMAP
      VtAinvMAV[nvec+1:spdim, 1:nvec] = VtAinvMAV[1:nvec, nvec+1:spdim]'
      VtAinvMAV[nvec+1:spdim, nvec+1:spdim] = AP'invMAP
      VtAinvMAV = Symmetric(VtAinvMAV)
    
      VtAV = Array{T,2}(undef, spdim, spdim)      
      VtAV[1:nvec, 1:nvec] = W2tAW2
      VtAV[1:nvec, nvec+1:spdim] .= 0
      VtAV[1:nvec, nvec+1:spdim] .= 0
      VtAV[nvec+1:spdim, nvec+1:spdim] .= 0
      for ivec in 1:spdim-nvec
        #VtAV[nvec+ivec, nvec+ivec] = P[:, ivec]'AP[:, ivec] # bugs inexplicably
        VtAV[nvec+ivec, nvec+ivec] = P[:, ivec]'A*P[:, ivec]
      end
      VtAV = Symmetric(VtAV)
    
      vals, vecs = eigen(VtAinvMAV, VtAV)

      W2temp .= W2
      for ivec in 1:nvec
        W2[:, ivec] = W2temp * vecs[1:nvec, ivec] .+ P * vecs[nvec+1:spdim, ivec]
      end

      restarted_once = true
    end
  end

  return x, it, res_norm[1:it], W2
end


"""
     trhrpcg(A::Union{SparseMatrixCSC{T},
                      FunctionMap},
             b::Array{T,1},
             x::Array{T,1},
             M
             nvec::Int,
             spdim::Int) 

Performs TR-RR-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M. 
Returns basis of new deflation subspaces computed by TR-HR projection.

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
function trhrpcg(A::Union{SparseMatrixCSC{T},
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
    if !restarted_once
      ivec = it
    else
      ivec = (it - spdim) % (spdim - nvec)
      ivec == 0 ? ivec = spdim - nvec : nothing
    end
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
        
      vals, vecs = eigen(VtAinvMAV, VtAV)
      for ivec in 1:nvec
        W2[:, ivec] = P * vecs[1:spdim, ivec]
      end

      restarted_once = true

    elseif restarted_once && ivec == spdim - nvec

      if isa(A, FunctionMap)
        for ivec in 1:nvec
          W2tA[ivec, :] .= A * W2[:, ivec]
        end
      else
        mul!(W2tA, W2', A)
      end
      W2tAW2 = W2tA * W2
      W2tAW2 = Symmetric(W2tAW2)

      for ivec in 1:nvec
        invMAW2[:, ivec] = M \ W2tA[ivec, :]
      end
      if isa(A, FunctionMap)
        for ivec in 1:(spdim-nvec)
          AP[:, ivec] .= A * P[:, ivec]
        end
      else
        mul!(AP[:, 1:spdim-nvec], A, P[:, 1:spdim-nvec])
      end
      for ivec in 1:spdim-nvec
        invMAP[:, ivec] = M \ AP[:, ivec]
      end
      VtAinvMAV = Array{T,2}(undef, spdim, spdim)
      VtAinvMAV[1:nvec, 1:nvec] = W2tA * invMAW2
      VtAinvMAV[1:nvec, nvec+1:spdim] = W2tA * invMAP[:, 1:spdim-nvec]
      VtAinvMAV[nvec+1:spdim, 1:nvec] = VtAinvMAV[1:nvec, nvec+1:spdim]'
      VtAinvMAV[nvec+1:spdim, nvec+1:spdim] = AP[:, 1:spdim-nvec]'invMAP[:, 1:spdim-nvec]
      VtAinvMAV = Symmetric(VtAinvMAV)
    
      VtAV = Array{T,2}(undef, spdim, spdim)
      VtAV[1:nvec, 1:nvec] = W2tAW2
      VtAV[1:nvec, nvec+1:spdim] .= 0
      VtAV[1:nvec, nvec+1:spdim] .= 0
      VtAV[nvec+1:spdim, nvec+1:spdim] .= 0
      for ivec in 1:spdim-nvec
        #VtAV[nvec+ivec, nvec+ivec] = P[:, ivec]'AP[:, ivec] # bugs inexplicably
        VtAV[nvec+ivec, nvec+ivec] = P[:, ivec]'A*P[:, ivec]
      end
      VtAV = Symmetric(VtAV)

      W2temp .= W2
      vals, vecs = eigen(VtAinvMAV, VtAV)
      for ivec in 1:nvec
        W2[:, ivec] = W2temp * vecs[1:nvec, ivec] .+ P[:, 1:spdim-nvec] * vecs[nvec+1:spdim, ivec]
      end
    end

  end

  return x, it, res_norm[1:it], W2
end