function add_my_procs(machines::Array{String,1},
                      n_local_proc::Int)

  for machine in machines
    if machine in ("hector", "lucien", "marcel", "andrew")
      addprocs(([("nicolas@$machine", :auto)]), 
               tunnel=true,
               dir="/home/nicolas/Dropbox/Git/julia-fem/",
               exename="/home/nicolas/julia-1.5.3/bin/julia",
               topology=:master_worker)

    elseif machine == "moorcock"
      addprocs(([("venkovic@moorcock", :auto)]), 
               tunnel=true,
               dir="/home/venkovic/Dropbox/Git/julia-fem/",
               exename="/home/venkovic/julia-1.5.3/bin/julia",
               topology=:master_worker)
    end
  end

  # Add local procs after remote procs to avoid issues with ClusterManagers
  addprocs(n_local_proc, topology=:master_worker) 
  
end
