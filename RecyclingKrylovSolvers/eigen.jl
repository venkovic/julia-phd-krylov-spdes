function lanczos(A::Union{SparseMatrixCSC{T},
                          FunctionMap},
                 nev::Int,
                 nvec::Int,
                 which=:MD)
  
  if typeof(A) == SparseMatrixCSC{Float64,Int}
    n = A.n
  else
    n = A.N
  end
  V = Array{Float64,2}(undef, n, nvec)
  w = Vector{Float64}(undef, n)
  p = Vector{Float64}(undef, n)
  T = Tridiagonal(zeros(nvec-1), zeros(nvec), zeros(nvec-1))
  Y = Array{Float64,2}(undef, n, nev)
  vals = Vector{Float64}(undef, nev)
  res = Vector{Float64}(undef, nev)

  V[:, 1] .= rand(n)
  V[:, 1] ./= sqrt(V[:, 1]'V[:, 1])

  p .= A * V[:, 1]
  α = V[:, 1]'p
  T[1, 1] = α
  β = 0.
  for i in 1:nvec-1
    if i == 1
      w .= p .- α * V[:, i]
    else
      w .= p .- α * V[:, i] .- β * V[:, i-1]
    end
    β = sqrt(w'w)
    V[:, i+1] .= w / β
    p .= A * V[:, i+1]
    α = V[:, i+1]'p
    T[i+1, i+1] = α
    T[i, i+1] = β
    T[i+1, i] = β
  end

  w .= p .- α * V[:, nvec] .- β * V[:, nvec-1]
  β = sqrt(w'w)

  T = SymTridiagonal(T)
  eigvals, eigvecs = eigen(T)

  if which == :MD
    vals .= reverse(eigvals)[1:nev]
    for i = 1:nev
      Y[:, i] = V * eigvecs[:, nvec-i+1]
      res[i] = abs(β * eigvecs[nvec, nvec-i+1])
    end

  elseif which == :LD
    vals .= eigvals[1:nev]
    for i = 1:nev
      Y[:, i] = V * eigvecs[:, i]
      res[i] = abs(β * eigvecs[nvec, i])
    end
  end

  return vals, Y, res
end