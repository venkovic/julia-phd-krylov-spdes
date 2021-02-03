module MyPreconditioners

using LinearAlgebra: cholesky
using SparseArrays: SparseMatrixCSC
using SuiteSparse.CHOLMOD: Factor

export BJPreconditioner
export Chol32Preconditioner
export Chol16Preconditioner

const T = Float64
const T32 = Float32
const T16 = Float16

include("BJPreconditioner.jl")
include("CholPreconditioners.jl")

end
