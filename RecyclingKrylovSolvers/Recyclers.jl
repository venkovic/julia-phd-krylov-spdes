using SparseArrays: SparseMatrixCSC

struct Recycler
  dims::Array{Int,1}
  W::Array{Float64,2}
end

prepare_recycler(A::SparseMatrixCSC{Float64,Int},
                 max_nvec::Int)

  W = Array{Float64,2}(undef, A.n, max_nvec)
  return Recycler([A.n, 0], W)
end