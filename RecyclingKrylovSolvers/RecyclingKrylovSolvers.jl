module RecyclingKrylovSolvers

using LinearAlgebra: mul!, axpy!, axpby!, dot, norm2, iszero, Eigen
using LinearAlgebra: Symmetric, eigvecs, eigen, svd, rank, diagm
using LinearAlgebra: Tridiagonal, SymTridiagonal, I, eigvals, Symmetric
using SparseArrays: SparseMatrixCSC
using LinearMaps: FunctionMap
import JLD

export cg, pcg
export eigcg, eigpcg
export defcg, eigdefcg, defpcg, eigdefpcg
export initcg, initpcg
export rrdefpcg, rrpcg, hrdefpcg, hrpcg
export trrrdefpcg, trrrpcg, trhrdefpcg, trhrpcg
export lotrrrdefpcg, lotrrrpcg, lotrhrdefpcg, lotrhrpcg
export Recycler, prepare_recycler
export lanczos

const T = Float64
const eps = 1e-7

include("cg.jl")
include("eigcg.jl")
include("defcg.jl")
include("initcg.jl")
include("rrdefpcg.jl")
include("hrdefpcg.jl")
include("trrrdefpcg.jl")
include("trhrdefpcg.jl")
include("lotrrrdefpcg.jl")
include("lotrhrdefpcg.jl")
include("Recyclers.jl")
include("eigen.jl")

end
