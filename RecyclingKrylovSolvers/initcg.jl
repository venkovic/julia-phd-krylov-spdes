"""
     initcg(A::Union{SparseMatrixCSC{T},
                     FunctionMap},
            b::Array{T,1},
            x::Array{T,1},
            W::Array{T,2})

Performs Init-CG (Erhel & Guyomarc'h, 2000).

Used to solve A x = b with an SPD matrix A when a set of linearly independent
vectors w1, w2, ... is known and such that Span{w1, w2, ...} is "approximately"
invariant under the action of A. Then an initial guess may be
generated which is deflated of the solution projected onto the invariant
subspace. Initializing a regular CG solve with such a deflated initial guess can
result in improvements of the convergence behavior.

Erhel, J. & Guyomarc'h, F.
An augmented conjugate gradient method for solving consecutive symmetric
positive definite linear systems,
SIAM Journal on Matrix Analysis and Applications, SIAM, 2000, 21, 1279-1299.

Giraud, L.; Ruiz, D. & Touhami, A.
A comparative study of iterative solvers exploiting spectral information
for SPD systems,
SIAM Journal on Scientific Computing, SIAM, 2006, 27, 1760-1786.

"""
function initcg(A::Union{SparseMatrixCSC{T},
                         FunctionMap},
                b::Array{T,1},
                x::Array{T,1},
                W::Array{T,2})

  n, nvec = size(W)
  r = Array{T,1}(undef,n)
  Ap = Array{T,1}(undef,n)
  res_norm = Array{T,1}(undef,n)
  p = Array{T,1}(undef,n)
  mu = Array{T,1}(undef,nvec)

  WtA = W' * A
  WtAW = WtA * W

  r .= b .- A * x
  mu .= W' * r
  mu .= WtAW \ mu
  x .+= W * mu

  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  p .= r
  res_norm[it] = sqrt(rTr)

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
    axpby!(1, r, beta, p) # p = beta * p + r
    it += 1
    res_norm[it] = sqrt(rTr)
  end

  return x, it, res_norm[1:it]
end


"""
initpcg(A::Union{SparseMatrixCSC{T},
                 FunctionMap},
        b::Vector{T},
        x::Vector{T},
        M,
        W::Array{T,2})
        
Performs Init-PCG (Erhel & Guyomarc'h, 2000).

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M when a
set of linearly independent vectors w1, w2, ... is known and such that
Span{w1, w2, ...} is "approximately" invariant under the action of M^{-1}A. Then
an initial guess may be generated which is deflated of the solution projected
onto the invariant subspace. Initializing a regular PCG solve with such a
deflated initial guess can result in improvements of the convergence behavior.

Erhel, J. & Guyomarc'h, F.
An augmented conjugate gradient method for solving consecutive symmetric
positive definite linear systems,
SIAM Journal on Matrix Analysis and Applications, SIAM, 2000, 21, 1279-1299.

Giraud, L.; Ruiz, D. & Touhami, A.
A comparative study of iterative solvers exploiting spectral information
for SPD systems,
SIAM Journal on Scientific Computing, SIAM, 2006, 27, 1760-1786.

"""
function initpcg(A::Union{SparseMatrixCSC{T},
                          FunctionMap},
                 b::Vector{T},
                 x::Vector{T},
                 M,
                 W::Array{T,2})

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

  it = 1
  r .= b .- A * x
  rTr = dot(r, r)
  z .= M \ r
  rTz = dot(r, z)
  p .= z
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