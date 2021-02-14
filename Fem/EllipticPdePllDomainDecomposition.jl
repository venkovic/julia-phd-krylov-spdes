"""
function pll_apply_schur(A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                         A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                         A_ΓΓ::SparseMatrixCSC{Float64,Int},
                         x::Array{Float64,1},
                         Π_IId;
                         verbose=true)

ndom = length(A_IId)

Sx = @sync @distributed (+) for idom in 1:ndom
  v = IterativeSolvers.cg(A_IId[idom], A_IΓd[idom] * x, Pl=Π_IId[idom])
  A_IΓd[idom]' * v
end
Sx .+= A_ΓΓ * x

return Sx
end
"""