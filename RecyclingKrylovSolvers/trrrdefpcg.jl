"""
     trrrdefpcg(A::Union{SparseMatrixCSC{T},
                         FunctionMap},
                b::Array{T,1},
                W::Array{T,2},
                x::Array{T,1},
                M) 

Performs TR-RR-Deflated-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M 
along with a deflation subspace Span{w1, w2, ...} spanned by the linearly 
independent vectors w1:=W[:,1], w2:=W[:,2], ... Returns basis of new 
deflation subspaces computed by TR-RR projection.

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
function trrrdefpcg(A::Union{SparseMatrixCSC{T},
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
    Z[:, ivec] = z ./ rTz^.5
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

      VtAV = Array{T,2}(undef, spdim, spdim)
      VtAV[1:nvec, 1:nvec] = W2tAW2
      VtAV[1:nvec, nvec+1:spdim] = W2tA * Z
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
      
      VtMV = Array{T,2}(undef, spdim, spdim)
      VtMV[1:nvec, 1:nvec] = I(nvec)
      VtMV[1:nvec, nvec+1:spdim] .= 0
      VtMV[1:nvec, nvec+1:spdim] .= 0
      VtMV[nvec+1:spdim, nvec+1:spdim] = I(spdim-nvec)
      VtMV = Symmetric(VtMV)
   
      W2temp .= W2
      vals, vecs = eigen(VtAV, VtMV)
      for ivec in 1:nvec
        W2[:, ivec] = W2temp * vecs[1:nvec, ivec] .+ Z * vecs[nvec+1:spdim, ivec]
      end

      restarted_once = true
    end
  end

  return x, it, res_norm[1:it], W2
end


"""
     trrrpcg(A::Union{SparseMatrixCSC{T},
                      FunctionMap},
             b::Array{T,1},
             x::Array{T,1},
             M
             nvec::Int,
             spdim::Int) 

Performs TR-RR-PCG.

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M. 
Returns basis of new deflation subspaces computed by TR-RR projection.

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
function trrrpcg(A::Union{SparseMatrixCSC{T},
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
        
      vals, vecs = eigen(VtAV, VtMV)
      for ivec in 1:nvec
        W2[:, ivec] = Z * vecs[1:spdim, ivec]
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

      VtAV = Array{T,2}(undef, spdim, spdim)
      VtAV[1:nvec, 1:nvec] .= W2tAW2
      VtAV[1:nvec, nvec+1:spdim] = W2tA * Z[:, 1:spdim-nvec]
      VtAV[nvec+1:spdim, 1:nvec] = VtAV[1:nvec, nvec+1:spdim]'
      if isa(A, FunctionMap)
        for ivec in 1:(spdim-nvec)
          AZ[:, ivec] .= A * Z[:, ivec]
        end
      else
        mul!(AZ[:, 1:spdim-nvec], A, Z[:, 1:spdim-nvec])
      end
      VtAV[nvec+1:spdim, nvec+1:spdim] = Z[:, 1:spdim-nvec]'AZ[:, 1:spdim-nvec]
      VtAV = Symmetric(VtAV)
 
      VtMV = Array{T,2}(undef, spdim, spdim)
      VtMV[1:nvec, 1:nvec] = I(nvec)
      VtMV[1:nvec, nvec+1:spdim] .= 0
      VtMV[1:nvec, nvec+1:spdim] .= 0
      VtMV[nvec+1:spdim, nvec+1:spdim] = I(spdim-nvec)
      VtMV = Symmetric(VtMV)
   
      W2temp .= W2
      vals, vecs = eigen(VtAV, VtMV)
      for ivec in 1:nvec
        W2[:, ivec] = W2temp * vecs[1:nvec, ivec] .+ Z[:, 1:spdim-nvec] * vecs[nvec+1:spdim, ivec]
      end
    end    
  end

  return x, it, res_norm[1:it], W2
end