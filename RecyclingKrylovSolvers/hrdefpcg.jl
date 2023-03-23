"""
     hrdefpcg(A::Union{SparseMatrixCSC{T},
                       FunctionMap},
              b::Array{T,1},
              W::Array{T,2},
              x::Array{T,1},
              M) 

Performs HR-Deflated-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M 
along with a deflation subspace Span{w1, w2, ...} spanned by the linearly 
independent vectors w1:=W[:,1], w2:=W[:,2], ... Returns basis of new 
deflation subspaces computed by HR projection.

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
function hrdefpcg(A::Union{SparseMatrixCSC{T},
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
  invMAP = Array{T,2}(undef, n, spdim-nvec)
  VtAV = Array{T,2}(undef, spdim, spdim)
  W2 = Array{T,2}(undef, n, nvec)

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
  P[:, it] = p
  
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
    if it <= (spdim - nvec)
      P[:, it] = p
    end
  end



  for ivec in 1:nvec
    invMAW[:, ivec] = M \ WtA[ivec, :]
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
  VtAinvMAV[1:nvec, 1:nvec] = WtA * invMAW
  VtAinvMAV[1:nvec, nvec+1:spdim] = WtA * invMAP
  VtAinvMAV[nvec+1:spdim, 1:nvec] = VtAinvMAV[1:nvec, nvec+1:spdim]'
  VtAinvMAV[nvec+1:spdim, nvec+1:spdim] = AP'invMAP
  VtAinvMAV = Symmetric(VtAinvMAV)

  VtAV[1:nvec, 1:nvec] = WtAW
  VtAV[1:nvec, nvec+1:spdim] .= 0
  VtAV[1:nvec, nvec+1:spdim] .= 0
  VtAV[nvec+1:spdim, nvec+1:spdim] .= 0
  for ivec in 1:spdim-nvec
    VtAV[nvec+ivec, nvec+ivec] = P[:, ivec]'AP[:, ivec]
  end
  VtAV = Symmetric(VtAV)

  vals, vecs = eigen(VtAinvMAV, VtAV)
  for ivec in 1:nvec
    W2[:, ivec] = W * vecs[1:nvec, ivec] .+ P * vecs[nvec+1:spdim, ivec]
  end

  return x, it, res_norm[1:it], W2
end


"""
     hrpcg(A::Union{SparseMatrixCSC{T},
                    FunctionMap},
           b::Array{T,1},
           x::Array{T,1},
           M
           nvec::Int,
           spdim::Int) 

Performs RR-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M. 
Returns basis of new deflation subspaces computed by HR projection.

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
function hrpcg(A::Union{SparseMatrixCSC{T},
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
  VtAV = Array{T,2}(undef, spdim, spdim)
  W2 = Array{T,2}(undef, n, nvec)

  maxit == 0 ? maxit = n : nothing
  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  res_norm[it] = sqrt(rTr)
  z .= M \ r
  rTz = dot(r, z)
  p .= z
  P[:, it] = p
  
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
    if it <= spdim
      P[:, it] = p
    end
  end

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

  return x, it, res_norm[1:it], W2
end