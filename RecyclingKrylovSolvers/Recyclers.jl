using SparseArrays: SparseMatrixCSC

struct Recycler
  W::Array{Float64,2}
end

function prepare_recycler(A::SparseMatrixCSC{Float64,Int},
                          nvec::Int)

  W = Array{Float64,2}(undef, A.n, nvec)
  return Recycler(W)
end