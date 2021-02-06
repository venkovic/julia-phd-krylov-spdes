struct Cholesky16 
  chol::SuiteSparse.CHOLMOD.Factor{T}
end

function get_cholesky16(A::SparseMatrixCSC{T})
  A = SparseMatrixCSC{T16,Int64}(A)
  chol = cholesky(A)
  #return chol
  return Cholesky16(chol)
end

import Base: \
function (\)(Πchol16::Cholesky16, x::Array{Float64,1})
  return Πchol16.chol \ Array{Float16,1}(x)
end

function LinearAlgebra.ldiv!(z::Array{Float64,1}, 
                             Πchol16::Cholesky16,
                             r::Array{Float64,1})
  z .= Πchol16.chol \ Array{Float16,1}(r)
end

function LinearAlgebra.ldiv!(Πchol16::Cholesky16,
                             r::Array{Float64,1})
  r .= Πchol16.chol \ Array{Float16,1}(r)
end


struct Cholesky32
  chol::SuiteSparse.CHOLMOD.Factor{T}
end

function get_cholesky32(A::SparseMatrixCSC{T})
  A = SparseMatrixCSC{T32,Int64}(A)
  chol = cholesky(A)
  #return chol
  return Cholesky32(chol)
end

import Base: \
function (\)(Πchol32::Cholesky32, x::Array{Float64,1})
  return Πchol32.chol \ Array{Float32,1}(x)
end

function LinearAlgebra.ldiv!(z::Array{Float64,1}, 
                             Πchol32::Cholesky32,
                             r::Array{Float64,1})
  z .= Πchol32.chol \ Array{Float32,1}(r)
end

function LinearAlgebra.ldiv!(Πchol32::Cholesky32,
                             r::Array{Float64,1})
  r .= Πchol32.chol \ Array{Float32,1}(r)
end

