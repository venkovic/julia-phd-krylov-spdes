module Utils
  
  using SparseArrays: SparseMatrixCSC, findnz, sparse
  import JLD

  # from PrintUtils.jl,
  export printlnln
  export space_println

  # from DeflationUtils.jl,
  export save_system, load_system
  export save_deflated_system

  include("PrintUtils.jl")
  include("DeflationUtils.jl")

end 