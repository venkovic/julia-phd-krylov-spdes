using Distributed

addprocs(([("marcel", 4)]), tunnel=true)
addprocs(([("andrew", 3)]), tunnel=true)
#addprocs(([("moorcock", 4)]), tunnel=true)
addprocs(3)

@everywhere begin
  push!(LOAD_PATH, "./Fem/")
  import Pkg
  Pkg.activate(".")
end

@everywhere begin 
  using Fem
  using Distributed
  using Distributions
  using DistributedOperations
end

import LinearAlgebra
using NPZ

