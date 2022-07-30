using Distributed

addprocs(([("hector0", 1)]), tunnel=true)
addprocs(([("hector1", 1)]), tunnel=true)
addprocs(([("hector2", 1)]), tunnel=true)

@everywhere begin
  import Pkg
  Pkg.activate(".")
  Pkg.resolve()
  Pkg.instantiate()
end
