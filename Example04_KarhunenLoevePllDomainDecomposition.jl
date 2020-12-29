using Distributed

addprocs(2)
addprocs(([("marcel", 1)]), tunnel=true)
addprocs(([("andrew", 1)]), tunnel=true)

println(nworkers())