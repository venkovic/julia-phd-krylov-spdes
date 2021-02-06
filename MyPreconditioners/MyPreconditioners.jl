module MyPreconditioners

import LinearAlgebra
using LinearAlgebra: cholesky
using SparseArrays: SparseMatrixCSC
import SuiteSparse

export BJPreconditioner

export Cholesky16, get_cholesky16
export Cholesky32, get_cholesky32

const T = Float64
const T32 = Float32
const T16 = Float16

include("BJPreconditioner.jl")
include("CholPreconditioners.jl")

end
