"""
cg(A, b, x)

Performs CG (Saad, 2003).

Saad, Y.
Iterative methods for sparse linear systems
SIAM, 2003, 82

# Examples
```jldoctest
julia>
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: cg;
const n = 1_000_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .05 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
const b = rand(T, n);
x = zeros(T, n);
x, it, res_norm = cg(A, b, x);
```
"""
function cg(A::Union{SparseMatrixCSC{T},
                     FunctionMap}, 
            b::Vector{T};
            x=nothing)


  isa(A, SparseMatrixCSC) ? n = A.n : n = A.N
  isnothing(x) ? x = zeros(n) : nothing
  r, Ap, res_norm = similar(x), similar(x), similar(x)
  #
  it = 1
  r[:] = b - A * x
  rTr = dot(r, r)
  p = copy(r)
  res_norm[it] = sqrt(rTr)
  bnorm = norm2(b)
  tol = eps * bnorm
  #
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
pcg(A, b, x, M)

Performs PCG (Saad, 2003).

Saad, Y.
Iterative methods for sparse linear systems
SIAM, 2003, 82.

# Examples
```jldoctest
julia>
using SparseArrays: spdiagm
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: pcg;
push!(LOAD_PATH, "./MyPreconditioners");
using MyPreconditioners: BJPreconditioner;
const n = 5_000_000;
const T = Float64;
const d0 = 3.5 .+ .05 * rand(T, n);
const d1 = -1 .+ .05 * rand(T, n - 1);
const d2 = -.5 .+ .05 * rand(T, n - 2);
const d3 = -.25 .+ .05 * rand(T, n - 3);
A = spdiagm(0 => d0, 1 => d1, 2 => d2, 3 => d3);
A = A .+ A';
const nblock = 10_000;
const M = BJPreconditioner(nblock, A);
const b = rand(T, n);
x = zeros(T, n);
x, it, res_norm = pcg(A, b, x, M);
```
"""
function pcg(A::Union{SparseMatrixCSC{T},
                      FunctionMap}, 
             b::Vector{T};
             x=nothing,
             M=nothing)
  
  isa(A, SparseMatrixCSC) ? n = A.n : n = A.N
  isnothing(x) ? x = zeros(n) : nothing
  r, z, Ap, res_norm = similar(x), similar(x), similar(x), similar(x)
  #
  it = 1
  r[:] = b - A * x
  rTr = dot(r, r)
  z[:] = M \ r
  rTz = dot(r, z)
  p = copy(z)
  res_norm[it] = sqrt(rTr)
  bnorm = norm2(b)
  tol = eps * bnorm
  #
  while (it < n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    z[:] = M \ r
    rTz = dot(r, z)
    beta *= rTz
    axpby!(1, z, beta, p) # p = beta * p + z
    it += 1
    res_norm[it] = sqrt(rTr)
  end
  return x, it, res_norm[1:it]
end
