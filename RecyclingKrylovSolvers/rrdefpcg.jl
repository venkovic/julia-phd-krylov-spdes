"""
     rrdefpcg(A::Union{SparseMatrixCSC{T},
                       FunctionMap},
              b::Array{T,1},
              W::Array{T,2},
              x::Array{T,1},
              M) 

Performs RR-Deflated-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M 
along with a deflation subspace Span{w1, w2, ...} spanned by the linearly 
independent vectors w1:=W[:,1], w2:=W[:,2], ... Returns basis of new 
deflation subspaces computed by RR projection.

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
function rrdefpcg(A::Union{SparseMatrixCSC{T},
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
  Z[:, it] = z ./ rTz^.5
  
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
      Z[:, it] = z ./ rTz^.5
    end
  end

  VtAV[1:nvec, 1:nvec] = WtAW
  VtAV[1:nvec, nvec+1:spdim] = WtA * Z
  VtAV[nvec+1:spdim, 1:nvec] = VtAV[1:nvec, nvec+1:spdim]'
  if isa(A, FunctionMap)
    for ivec in 1:(spdim-nvec)
      AZ[:, ivec] .= A * Z[:, ivec]
    end
  else
    mul!(AZ, A, Z)
  end
  VtAV[nvec+1:spdim, nvec+1:spdim] = Z'AZ
  VtAV = Symmetric(VtAV)

  VtMV[1:nvec, 1:nvec] = I(nvec)
  VtMV[1:nvec, nvec+1:spdim] .= 0
  VtMV[1:nvec, nvec+1:spdim] .= 0
  VtMV[nvec+1:spdim, nvec+1:spdim] = I(spdim-nvec)
  VtMV = Symmetric(VtMV)

  vals, vecs = eigen(VtAV, VtMV)
  for ivec in 1:nvec
    W2[:, ivec] = W * vecs[1:nvec, ivec] .+ Z * vecs[nvec+1:spdim, ivec]
  end

  return x, it, res_norm[1:it], W2
end


"""
     rrpcg(A::Union{SparseMatrixCSC{T},
                    FunctionMap},
           b::Array{T,1},
           x::Array{T,1},
           M
           nvec::Int,
           spdim::Int) 

Performs RR-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M. 
Returns basis of new deflation subspaces computed by RR projection.

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
function rrpcg(A::Union{SparseMatrixCSC{T},
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

  maxit == 0 ? maxit = n : nothing
  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  res_norm[it] = sqrt(rTr)
  z .= M \ r
  rTz = dot(r, z)
  p .= z
  Z[:, it] = z ./ rTz^.5
  
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
      Z[:, it] = z ./ rTz^.5
    end
  end

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

  vals, vecs = eigen(VtAV, VtMV)
  for ivec in 1:nvec
    W2[:, ivec] = Z * vecs[1:spdim, ivec]
  end

  return x, it, res_norm[1:it], W2
end