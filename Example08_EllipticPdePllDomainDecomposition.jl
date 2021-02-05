using Distributed

addprocs(([("lucien", :auto)]), tunnel=true, topology=:master_worker)
addprocs(([("hector", :auto)]), tunnel=true, topology=:master_worker)
#addprocs(([("marcel", :auto)]), tunnel=true, topology=:master_worker)
#addprocs(([("andrew", :auto)]), tunnel=true, topology=:master_worker)
#addprocs(([("celine", :auto)]), tunnel=true, topology=:master_worker)
#addprocs(([("venkovic@moorcock", :auto)]), tunnel=true,
#             dir="/home/venkovic/Dropbox/Git/julia-fem/",
#             exename="/home/venkovic/julia-1.5.3/bin/julia",
#             topology=:master_worker)
addprocs(Sys.CPU_THREADS - 2, topology=:master_worker) # Add local procs after remote procs to avoid issues with ClusterManagers

@everywhere push!(LOAD_PATH, "./Fem/")

@everywhere begin
  import Pkg
  Pkg.activate(".")
end

push!(LOAD_PATH, "./Utils/")
using Utils: space_println, printlnln

@everywhere begin 
  using Fem
  using Distributed
end

@everywhere worker_timeout() = 10_000.

using NPZ: npzwrite










































