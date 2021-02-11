module Utils
  
  using Distributed
  using DataStructures: Queue, enqueue!, dequeue!,
                        Stack, push!, pop!
  using SparseArrays: SparseMatrixCSC, findnz, sparse
  import JLD

  # from PrintUtils.jl,
  export printlnln
  export space_println

  # from DeflationUtils.jl,
  export save_system, load_system
  export save_deflated_system

  # from PllUtils.jl
  export add_my_procs
  export dynamic_mapreduce!

  include("PrintUtils.jl")
  include("DeflationUtils.jl")
  include("PllUtils.jl")

end 