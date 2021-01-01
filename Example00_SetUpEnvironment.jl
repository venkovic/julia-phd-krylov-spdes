using Distributed
addprocs(([("marcel", 1)]), tunnel=true)
addprocs(([("andrew", 1)]), tunnel=true)

@everywhere begin
  import Pkg
  Pkg.instantiate()
end