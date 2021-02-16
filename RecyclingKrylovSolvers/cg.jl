"""
     cg(A::Union{SparseMatrixCSC{T},
                 FunctionMap}, 
        b::Array{T,1},
        x::Array{T,1})

Performs CG (Saad, 2003).

Saad, Y.
Iterative methods for sparse linear systems
SIAM, 2003, 82

"""
function cg(A::Union{SparseMatrixCSC{T},
                     FunctionMap}, 
            b::Array{T,1},
            x::Array{T,1})

  isa(A, SparseMatrixCSC) ? n = A.n : n = A.N

  r = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)

  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  copyto!(p, r)
  res_norm[it] = sqrt(rTr)
  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTr / d
    beta = 1. / rTr
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    beta *= rTr
    axpby!(1, r, beta, p) # p = beta * p + r
    it += 1
    res_norm[it] = sqrt(rTr)
  end

  return x, it, res_norm[1:it]
end


"""
     pcg(A::Union{SparseMatrixCSC{T},
                  FunctionMap},
         b::Array{T,1},
         x::Array{T,1},
         M)

Performs PCG (Saad, 2003).

Saad, Y.
Iterative methods for sparse linear systems
SIAM, 2003, 82.

"""
function pcg(A::Union{SparseMatrixCSC{T},
                      FunctionMap},
             b::Array{T,1},
             x::Array{T,1},
             M)
  
  isa(A, SparseMatrixCSC) ? n = A.n : n = A.N

  r = Array{T,1}(undef, n)
  z = Array{T,1}(undef, n)
  p = Array{T,1}(undef, n)
  Ap = Array{T,1}(undef, n)
  res_norm = Array{T,1}(undef, n)

  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  z .= M \ r
  rTz = dot(r, z)
  copyto!(p, z)
  res_norm[it] = sqrt(rTr)
  bnorm = norm2(b)
  tol = eps * bnorm

  while (it < n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    z .= M \ r
    rTz = dot(r, z)
    beta *= rTz
    axpby!(1, z, beta, p) # p = beta * p + z
    it += 1
    res_norm[it] = sqrt(rTr)
  end

  return x, it, res_norm[1:it]
end