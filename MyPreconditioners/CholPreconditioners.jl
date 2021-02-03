function Chol32Preconditioner(A::SparseMatrixCSC{T})
  A = SparseMatrixCSC{T32,Int64}(A)
  Chol = cholesky(A)
  return Chol
end

function Chol16Preconditioner(A::SparseMatrixCSC{T})
  A = SparseMatrixCSC{T16,Int64}(A)
  Chol = cholesky(A)
  return Chol
end
