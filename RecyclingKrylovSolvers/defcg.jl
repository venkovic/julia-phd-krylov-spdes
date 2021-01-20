"""
defcg(A, b, x, W)

Performs Deflated-CG (Saad et al., 2000).

Used to solve A x = b with an SPD matrix A when a set of linearly independent
vectors w1, w2, ... is known such that Span{w1, w2, ...} is "approximately"
invariant under the action of A. The sequence of iterates of Def-CG is
equivalent to a post-processed sequence of the regular CG solve of a deflated
version of the linear system, with guaranteed decrease of the condition
number. Remark: if Span{w1, w2, ...} is exactly invariant under the action of A,
one should use Init-CG instead of Def-CG because both algorithms would then have
equally positive impacts on convergence, but Def-CG requires an additional
computational cost at every solver iteration.

Saad, Y.; Yeung, M.; Erhel, J. & Guyomarc'h, F.
Deflated Version of the Conjugate Gradient Algorithm,
SIAM Journal on Scientific Computing, SIAM, 1999, 21, 1909-1926.

# Examples
```jldoctest
julia>
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: defcg, cg;
using Random: seed!
seed!(1234);
const n = 1_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
#
# Example: Fixed SPD A with multiple right-hand sides bs
function mrhs_defcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int)
  _, W = eigs(A; nev=nvec, which=:SM);
  println("\\n* Def-CG results *");
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itdefcg, _ = defcg(A, b, zeros(T, n), W);
    _, itcg, _ = cg(A, b, zeros(T, n));
    println("Def-CG: ", itdefcg, ", CG: ", itcg);
  end
end
nsmp, nvec = 5, 20;
mrhs_defcg(A, nvec, nsmp);

* Def-CG results *
Def-CG: 140, CG: 183
Def-CG: 141, CG: 183
Def-CG: 141, CG: 183
Def-CG: 140, CG: 184
Def-CG: 140, CG: 183
```
"""
function defcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2})
  r, Ap, res_norm, p = similar(x), similar(x), similar(x), similar(x)
  #
  WtA = W' * A
  WtAW = WtA * W
  #
  if iszero(x)
    r .= b
  else
    r = b - A * x
  end
  mu = W' * r
  mu = WtAW \ mu
  x += W * mu
  #
  it = 1
  r = b - A * x
  rTr = dot(r, r)
  mu = WtAW \ (WtA * r)
  p = r - (W * mu)
  res_norm[it] = sqrt(rTr)
  #
  bnorm = norm2(b)
  tol = eps * bnorm
  #
  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTr / d
    beta = 1. / rTr
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    beta *= rTr
    mu = WtAW \ (WtA * r)
    p = beta * p + r - (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
  end
  return x, it, res_norm[1:it]
end

"""
eigdefcg(A, b, x, W, spdim)

Performs RR-LO-TR-Def-CG (Venkovic et al., 2020), here referred to as eigDef-CG.

Works as a combination of eigCG and Def-CG. The linear solve is deflated as in
Def-CG, and approximate least dominant eigenvectors of A are computed throughout
the solve in a similar way as in eigCG. This algorithm is an alternative to
the incremental eigCG algorithm when solving for a sequence of systems A xs = bs
with a constant SPD matrix A and different right-hand sides bs. This algorithm
should be the method of choice when solving a sequence of linear systems of the
form As xs = bs with correlated SPD matrices A1, A2, ... Examples are shown
below for each type of problem.

Venkovic, N.; Mycek, P; Giraud, L.; Le Maître, O.
Recycling Krylov subspace strategiesfor sequences of sampled stochastic
elliptic equations,
SIAM Journal on Scientific Computing, SIAM, 2020, under review.

# Examples
```jldoctest
julia>
using LinearAlgebra: SymTridiagonal;
using SparseArrays: sparse, SparseMatrixCSC;
using Arpack: eigs;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: eigcg, eigdefcg, defcg, cg, initcg;
using Random: seed!
seed!(1234);
const n = 1_000;
const T = Float64;
A = sparse(SymTridiagonal(2 .+ .5 * rand(T, n), -1 .+ .05 * rand(T, n-1)));
A = A * A;
nsmp, ndefcg, nvec, spdim = 10, 3, 20, 50;
#
# Example 1: Fixed SPD A with multiple right-hand sides bs
function mrhs_eigdefcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int, spdim::Int, ndefcg::Int)
  _, U = eigs(A; nev=nvec, which=:SM);
  W = Array{T}(undef, (n, nvec));
  println("\\n* eigDef-CG results for multiple right-hand sides *");
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itcg, _ = cg(A, b, zeros(T, n));
    _, itdefcg, _ = defcg(A, b, zeros(T, n), U);
    if ismp == 1
      _, iteigcg, _, W = eigcg(A, b, zeros(T, n), nvec, spdim);
      println("eigCG: ", iteigcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
    else
      if ismp <= ndefcg
        _, iteigdefcg, _, W = eigdefcg(A, b, zeros(T, n), W, spdim);
        println("eigDef-CG: ", iteigdefcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
      else
        _, itinitcg, _ = initcg(A, b, zeros(T, n), W);
        println("Init-CG: ", itinitcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
      end
    end
  end
end
mrhs_eigdefcg(A, nvec, nsmp, spdim, ndefcg);
#
# Example 2: Multiple SPD matrices As with a constant right-hand side b
function mops_eigdefcg(A::SparseMatrixCSC{T}, nvec::Int, nsmp::Int, spdim::Int)
  b = rand(T, n);
eigDef-CG: 155, Def-CG: 141, CG: 183
  W = Array{T}(undef, (n, nvec));
  println("\\n* eigDef-CG results for multiple operators *");
  for ismp in 1:nsmp
    A += sparse(SymTridiagonal(.12 * rand(T, n), -.05 * rand(T, n-1)));
    _, U = eigs(A; nev=nvec, which=:SM);
    _, itcg, _ = cg(A, b, zeros(T, n));
    _, itdefcg, _ = defcg(A, b, zeros(T, n), U);
    if ismp == 1
      _, iteigcg, _, W = eigcg(A, b, zeros(T, n), nvec, spdim);
      println("eigCG: ", iteigcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
    else
      _, iteigdefcg, _, W = eigdefcg(A, b, zeros(T, n), W, spdim);
      println("eigDef-CG: ", iteigdefcg, ", Def-CG: ", itdefcg, ", CG: ", itcg);
    end
  end
end
mops_eigdefcg(A, nvec, nsmp, spdim);

* eigDef-CG results for multiple right-hand sides *
eigCG: 183, Def-CG: 140, CG: 183
eigDef-CG: 155, Def-CG: 141, CG: 183
eigDef-CG: 153, Def-CG: 141, CG: 183
Init-CG: 161, Def-CG: 140, CG: 184
Init-CG: 160, Def-CG: 140, CG: 183
Init-CG: 159, Def-CG: 140, CG: 183
Init-CG: 155, Def-CG: 140, CG: 183
Init-CG: 161, Def-CG: 140, CG: 184
Init-CG: 156, Def-CG: 140, CG: 183
Init-CG: 162, Def-CG: 140, CG: 183

* eigDef-CG results for multiple operators *
eigCG: 196, Def-CG: 136, CG: 196
eigDef-CG: 176, Def-CG: 126, CG: 226
eigDef-CG: 152, Def-CG: 122, CG: 253
eigDef-CG: 150, Def-CG: 119, CG: 231
eigDef-CG: 127, Def-CG: 113, CG: 183
eigDef-CG: 113, Def-CG: 111, CG: 160
eigDef-CG: 112, Def-CG: 108, CG: 154
eigDef-CG: 112, Def-CG: 107, CG: 140
eigDef-CG: 111, Def-CG: 100, CG: 137
eigDef-CG: 103, Def-CG: 98, CG: 135
```
"""
function eigdefcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, W::Array{T,2}, spdim::Int)
  r, Ap, res_norm, p = similar(x), similar(x), similar(x), similar(x)
  #
  WtA = W' * A
  WtAW = WtA * W
  #
  if iszero(x)
    r .= b
  else
    r = b - A * x
  end
  mu = W' * r
  mu = WtAW \ mu
  x += W * mu
  #
  n = size(x)[1]
  nvec = size(W)[2]
  nev = nvec
  V = Array{T}(undef, (n, spdim))
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, (spdim, 2 * nvec))
  first_restart = true
  #
  it = 1
  r = b - A * x
  rTr = dot(r, r)
  mu = WtAW \ (WtA * r)
  p = r - (W * mu)
  res_norm[it] = sqrt(rTr)
  #
  VtAV[1:nvec, 1:nvec] = WtAW
  V[:, 1:nvec] = W
  #
  ivec = nvec + 1
  V[:, ivec] = r / res_norm[it]
  #
  bnorm = norm2(b)
  tol = eps * bnorm
  #
  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTr / d
    beta = 1. / rTr
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    beta *= rTr
    mu = WtAW \ (WtA * r)
    p = beta * p + r - (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
    #
    VtAV[ivec, ivec] += 1 / alpha
    #
    if ivec == spdim
      if first_restart
        VtAV[1:nvec, nvec+1:spdim] = WtA * V[:, nvec+1:spdim]
        first_restart = false
      end
      Tm = Symmetric(VtAV) # spdim-by-spdim
      Y[:, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # spdim-by-nvec
      Y[1:spdim-1, nvec+1:end] = eigvecs(Tm[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      vals, Z = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      V[:, 1:nev] = V * (Q * Z) # n-by-nev
      #
      ivec = nev + 1
      V[:, ivec] = r / res_norm[it]
      VtAV .= 0
      for j in 1:nev
        VtAV[j, j] = vals[j]
      end
      VtAV[ivec, ivec] = beta / alpha
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
defpcg(A, b, x, M, W)

Performs Deflated-PCG (Saad et al., 2000).

Used to solve A x = b with an SPD matrix A and an SPD preconditioner M, when a
set of linearly independent vectors w1, w2, ... is known such that
Span{w1, w2, ...} is "approximately" invariant under the action of M^{-1}A.
The sequence of iterates of Def-PCG is equivalent to a post-processed sequence
of the regular CG solve of a deflated and split-preconditioned version of the
linear system, with guaranteed decrease of the condition number.
Remark: if Span{w1, w2, ...} is exactly invariant under the action of M^{-1}A,
one should use Init-PCG instead of Def-PCG because both algorithms would then
have equally positive impacts on convergence, but Def-PCG requires an additional
computational cost at every solver iteration.

Saad, Y.; Yeung, M.; Erhel, J. & Guyomarc'h, F.
Deflated Version of the Conjugate Gradient Algorithm,
SIAM Journal on Scientific Computing, SIAM, 1999, 21, 1909-1926.

# Examples
```jldoctest
julia>
using LinearAlgebra: I;
using SparseArrays: SparseMatrixCSC, sprand;
using Arpack: eigs;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: defpcg, pcg;
push!(LOAD_PATH, "./MyPreconditioners");
using MyPreconditioners: BJPreconditioner;
using Random: seed!
seed!(1234);
const n = 5_000;
const T = Float64;
const nblock = 5;
A = sprand(T, n, n, .0001);
A += A' + 2 * I;
A = A * A;
M = BJPreconditioner(nblock, A);
#
# Example: Fixed SPD A with multiple right-hand sides bs
function mrhs_defpcg(A::SparseMatrixCSC{T}, M, nvec::Int, nsmp::Int)
  _, W = eigs(A; nev=nvec, which=:SM);
  println("\\n* Def-CG results *");
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itdefpcg, _ = defpcg(A, b, zeros(T, n), M, W);
    _, itpcg, _ = pcg(A, b, zeros(T, n), M);
    println("Def-PCG: ", itdefpcg, ", PCG: ", itpcg);
  end
end
nsmp, nvec = 5, 20;
mrhs_defpcg(A, M, nvec, nsmp);

* Def-PCG results *
Def-PCG: 69, PCG: 259
Def-PCG: 69, PCG: 260
Def-PCG: 68, PCG: 260
Def-PCG: 71, PCG: 258
Def-PCG: 70, PCG: 259
```
"""
function defpcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, M, W::Array{T,2})
  r, Ap, res_norm, p, z = similar(x), similar(x), similar(x), similar(x), similar(x)
  #
  WtA = W' * A
  WtAW = WtA * W
  #
  if iszero(x)
    r .= b
  else
    r = b - A * x
  end
  mu = W' * r
  mu = WtAW \ mu
  x += W * mu
  #
  it = 1
  r = b - A * x
  rTr = dot(r, r)
  z = (M \ r)::Vector{T}
  rTz = dot(r, z)
  mu = WtAW \ (WtA * z)
  p = z - (W * mu)
  res_norm[it] = sqrt(rTr)
  #
  bnorm = norm2(b)
  tol = eps * bnorm
  #
  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    rTr = dot(r, r)
    z = (M \ r)::Vector{T}
    rTz = dot(r, z)
    beta *= rTz
    mu = WtAW \ (WtA * z)
    p = beta * p + z - (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
  end
  return x, it, res_norm[1:it]
end

"""
eigdefpcg(A, b, x, M, W, spdim)

Performs RR-LO-TR-Def-PCG (Venkovic et al., 2020), here referred to as eigDef-PCG.

Works as a combination of eigPCG and Def-PCG. The linear solve is deflated as in
Def-PCG, and approximate least dominant right eigenvectors of M^{-1}A are
computed throughout the solve in a similar way as in eigPCG. This algorithm is
an alternative to the incremental eigPCG algorithm when solving for a sequence
of systems A xs = bs with constant SPD A and M, and different right-hand sides
bs. This algorithm should be the method of choice when solving a sequence of
linear systems of the form As xs = bs with correlated SPD matrices A1, A2, ...
Examples are shown below for each type of problem.

Venkovic, N.; Mycek, P; Giraud, L.; Le Maître, O.
Recycling Krylov subspace strategiesfor sequences of sampled stochastic
elliptic equations,
SIAM Journal on Scientific Computing, SIAM, 2020, under review.

# Examples
```jldoctest
julia>
using LinearAlgebra: I, SymTridiagonal;
using SparseArrays: SparseMatrixCSC, sprand;
using Arpack: eigs;
push!(LOAD_PATH, "./MyRecyclingKrylovSolvers");
using MyRecyclingKrylovSolvers: eigpcg, eigdefpcg, defpcg, pcg, initpcg;
push!(LOAD_PATH, "./MyPreconditioners");
using MyPreconditioners: BJPreconditioner;
using Random: seed!
seed!(1234);
const n = 5_000;
const T = Float64;
const nblock = 5;
A = sprand(T, n, n, .0001);
A += A' + 2 * I;
A = A * A;
M = BJPreconditioner(nblock, A);
nsmp, ndefcg, nvec, spdim = 10, 3, 5, 25;
#
# Example 1: Fixed SPD A with multiple right-hand sides bs
function mrhs_eigdefpcg(A::SparseMatrixCSC{T}, M, nvec::Int, nsmp::Int, spdim::Int, ndefcg::Int)
  _, U = eigs(A; nev=nvec, which=:SM);
  W = Array{T}(undef, (n, nvec));
  println("\\n* eigDef-PCG results for multiple right-hand sides *");
  for ismp in 1:nsmp
    b = rand(T, n);
    _, itpcg, _ = pcg(A, b, zeros(T, n), M);
    _, itdefpcg, _ = defpcg(A, b, zeros(T, n), M, U);
    if ismp == 1
      _, iteigpcg, _, W = eigpcg(A, b, zeros(T, n), M, nvec, spdim);
      println("eigPCG: ", iteigpcg, ", Def-PCG: ", itdefpcg, ", PCG: ", itpcg);
    else
      if ismp <= ndefcg
        _, iteigdefpcg, _, W = eigdefpcg(A, b, zeros(T, n), M, W, spdim);
        println("eigDef-PCG: ", iteigdefpcg, ", Def-PCG: ", itdefpcg, ", PCG: ", itpcg);
      else
        _, itinitpcg, _ = initpcg(A, b, zeros(T, n), M, W);
        println("Init-PCG: ", itinitpcg, ", Def-PCG: ", itdefpcg, ", PCG: ", itpcg);
      end
    end
  end
end
mrhs_eigdefpcg(A, M, nvec, nsmp, spdim, ndefcg);
#
# Example 2: Multiple SPD matrices As with a constant right-hand side b
function mops_eigdefpcg(A::SparseMatrixCSC{T}, M, nvec::Int, nsmp::Int, spdim::Int)
  b = rand(T, n);
  W = Array{T}(undef, (n, nvec));
  println("\\n* eigDef-PCG results for multiple operators *");
  for ismp in 1:nsmp
    A += .05 * SymTridiagonal(.12 * rand(T, n), -.05 * rand(T, n-1));
    _, U = eigs(A; nev=nvec, which=:SM);
    _, itpcg, _ = pcg(A, b, zeros(T, n), M);
    _, itdefpcg, _ = defpcg(A, b, zeros(T, n), M, U);
    if ismp == 1
      _, iteigpcg, _, W = eigpcg(A, b, zeros(T, n), M, nvec, spdim);
      println("eigPCG: ", iteigpcg, ", Def-PCG: ", itdefpcg, ", PCG: ", itpcg);
    else
      _, iteigdefpcg, _, W = eigdefpcg(A, b, zeros(T, n), M, W, spdim);
      println("eigDef-PCG: ", iteigdefpcg, ", Def-PCG: ", itdefpcg, ", PCG: ", itpcg);
    end
  end
end
mops_eigdefpcg(A, M, nvec, nsmp, spdim);

* eigDef-PCG results for multiple right-hand sides *
eigPCG: 259, Def-PCG: 126, PCG: 259
eigDef-PCG: 125, Def-PCG: 127, PCG: 260
eigDef-PCG: 123, Def-PCG: 124, PCG: 260
Init-PCG: 129, Def-PCG: 129, PCG: 258
Init-PCG: 138, Def-PCG: 127, PCG: 259
Init-PCG: 125, Def-PCG: 127, PCG: 260
Init-PCG: 125, Def-PCG: 125, PCG: 258
Init-PCG: 140, Def-PCG: 129, PCG: 263
Init-PCG: 127, Def-PCG: 128, PCG: 260
Init-PCG: 136, Def-PCG: 127, PCG: 262

* eigDef-PCG results for multiple operators *
eigPCG: 227, Def-PCG: 122, PCG: 227
eigDef-PCG: 114, Def-PCG: 116, PCG: 209
eigDef-PCG: 109, Def-PCG: 112, PCG: 196
eigDef-PCG: 107, Def-PCG: 111, PCG: 185
eigDef-PCG: 105, Def-PCG: 108, PCG: 177
eigDef-PCG: 102, Def-PCG: 103, PCG: 169
eigDef-PCG: 100, Def-PCG: 102, PCG: 160
eigDef-PCG: 98, Def-PCG: 99, PCG: 155
eigDef-PCG: 96, Def-PCG: 107, PCG: 145
eigDef-PCG: 95, Def-PCG: 95, PCG: 141
```
"""
function eigdefpcg(A::SparseMatrixCSC{T}, b::Vector{T}, x::Vector{T}, M, W::Array{T,2}, spdim::Int)
  r, Ap, res_norm, p, z = similar(x), similar(x), similar(x), similar(x), similar(x)
  #
  WtA = W' * A
  WtAW = WtA * W
  WtW = W'W
  #
  if iszero(x)
    r .= b
  else
    r = b - A * x
  end
  mu = W' * r
  mu = WtAW \ mu
  x += W * mu
  #
  n = size(x)[1]
  nvec = size(W)[2]
  nev = nvec
  V = Array{T}(undef, (n, spdim))
  VtAV = zeros(T, spdim, spdim)
  Y = zeros(T, (spdim, 2 * nvec))
  first_restart = true
  #
  it = 1
  r = b - A * x
  rTr = dot(r, r)
  z = (M \ r)::Vector{T}
  rTz = dot(r, z)
  mu = WtAW \ (WtA * z)
  p = z - (W * mu)
  res_norm[it] = sqrt(rTr)
  #
  VtAV[1:nvec, 1:nvec] = WtAW
  V[:, 1:nvec] = W
  just_restarted = false
  #
  ivec = nvec + 1
  V[:, ivec] = z / sqrt(rTz)
  #
  bnorm = norm2(b)
  tol = eps * bnorm
  #
  while (it < A.n) && (res_norm[it] > tol)
    mul!(Ap, A, p) # Ap = A * p
    d = dot(p, Ap)
    alpha = rTz / d
    beta = 1. / rTz
    axpy!(alpha, p, x) # x += alpha * p
    axpy!(-alpha, Ap, r) # r -= alpha * Ap
    r -= W * (WtW \ (W' * r))
    rTr = dot(r, r)
    z = (M \ r)::Vector{T}
    rTz = dot(r, z)
    beta *= rTz
    mu = WtAW \ (WtA * z)
    p = beta * p + z - (W * mu)
    it += 1
    res_norm[it] = sqrt(rTr)
    #
    VtAV[ivec, ivec] += 1 / alpha
    #
    if ivec == spdim
      if first_restart
        VtAV[1:nvec, nvec+1:spdim] = WtA * V[:, nvec+1:spdim]
        first_restart = false
      end
      Tm = Symmetric(VtAV) # spdim-by-spdim
      Y[:, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # spdim-by-nvec
      Y[1:spdim-1, nvec+1:end] = eigvecs(Tm[1:spdim-1, 1:spdim-1])[:, 1:nvec] # (spdim-1)-by-nvec
      nev = rank(Y) # nvec <= nev <= 2*nvec
      Q = svd(Y).U[:, 1:nev] # spdim-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      vals, Z = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      V[:, 1:nev] = V * (Q * Z) # n-by-nev
      #
      ivec = nev + 1
      V[:, ivec] = z / sqrt(rTz)
      VtAV .= 0
      for j in 1:nev
        VtAV[j, j] = vals[j]
      end
      VtAV[ivec, ivec] = beta / alpha
      just_restarted = true
    else
      just_restarted = false
      ivec += 1
      V[:, ivec] = z / sqrt(rTz)
      VtAV[ivec - 1, ivec] = - sqrt(beta) / alpha
      VtAV[ivec, ivec] = beta / alpha
    end
  end
  if !just_restarted
    if ivec > nvec
      ivec -= 1
      if first_restart
        VtAV[1:nvec, nvec+1:ivec] = WtA * V[:, nvec+1:ivec]
      end
      Tm = Symmetric(VtAV[1:ivec, 1:ivec]) # ivec-by-ivec
      Y .= 0
      Y[1:ivec, 1:nvec] = eigvecs(Tm)[:, 1:nvec] # ivec-by-nvec
      Y[1:ivec-1, nvec+1:end] = eigvecs(Tm[1:ivec-1, 1:ivec-1])[:, 1:nvec] # (ivec-1)-by-nvec
      nev = rank(Y[1:ivec, :]) # nvec <= nev <= 2*nvec
      Q = svd(Y[1:ivec, :]).U[:, 1:nev] # ivec-by-nev
      H = Q' * (Tm * Q) # nev-by-nev
      vals, Z = eigen(H)::Eigen{T,T,Array{T,2},Array{T,1}}
      V[:, 1:nev] = V[:, 1:ivec] * (Q * Z) # n-by-nev
    else
      println("Warning: Less CG iterations than the number of ",
              "eigenvectors wanted. Only Lanczos vectors may be returned.")
    end
  end
  return x, it, res_norm[1:it], V[:, 1:nvec]
end
