function add_my_procs(machines::Array{String,1},
                      n_local_proc::Int)

  for machine in machines
    if machine in ("rejean", "hector", "lucien", "marcel", "andrew")
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

  # Add local procs after remote procs to avoid issues with ClusterManagers.jl
  addprocs(n_local_proc, topology=:master_worker) 
  
end


"""
     function dynamic_mapreduce!(func::Function,
                                 redop::Function,
                                 coll::Array{Int,1},
                                 K::Array{Float64,2};
                                 verbose=true,
                                 Δt=2.)


Does parallel mapreduce of arrays with dynamic scheduling. This is an alternative to 

K .= @distributed (redop) for c in coll
  func(c)
end

which does static scheduling and tends to crash for time consuming and unbalanced work 
loads. 

Another approach is given by 

reduce(redop, Distributed.pmap(func, K))

which requires to allocate enough memory to store Distributed.pmap(func, K). Since func
is ***, this becomes a problem when the number of workers and the dimensions of K are 
increased. 

Input:

 `func::Function`, ``,
  function used to map the collection of parameters.

 `redop::Function`, 
  shape-preserving reduction operator, e.g., (+), ...

 `coll::Array{Int,1}`,
  approximate number of DoFs wanted.

 `K::Array{Float64,2}`,
  covariance function, must be available everywhere.

 `verbose=true`
  filename's root.

 `forget=1e-6`,
  threshold of covariance between points in distinct subdomains under which
  subdomain pairs are ignored for the assembly of the reduced global mass
  covariance matrix. Note that `forget<0` ⟹ all pairs are considered.

Output:

 `ind_Id_g2l::Array{Dict{Int,Int}}`, 
  conversion tables from global to local indices of nodes strictly inside each subdomain.

 `ind_Γd_g2l::Array{Dict{Int,Int}}`,
  conversion table from global to local indices of nodes on the interface of each subdomain.

"""
function dynamic_mapreduce!(func::Function,
                            redop::Function,
                            coll::Union{UnitRange{Int},  
                                        Array{Int,1}},
                            K::Array{Float64,2};
                            verbose=true,
                            Δt=2.)
 
  njobs = length(coll)
  
  pending_jobs_id = Queue{Int}()
  for job_id in 1:njobs 
    enqueue!(pending_jobs_id, job_id)
  end

  done_jobs_id = Stack{Int}()

  running_jobs_id = Dict{Int,Int}(worker => 0 for worker in workers())
  running_jobs = Dict{Int,Task}()

  while length(done_jobs_id) < njobs
    
    sleep(Δt)

    # Loop over running jobs
    for (worker, job_id) in running_jobs_id

      # Worker is free
      if (job_id == 0) && (length(pending_jobs_id) > 0)
        
        # Launch a new job
        new_job_id = dequeue!(pending_jobs_id)
        new_job = @async remotecall_fetch(func, worker, coll[new_job_id])
        
        # New job was successfully launched
        if new_job.state in (:runnable, :running, :done)
          running_jobs_id[worker] = new_job_id
          running_jobs[worker] = new_job
          if verbose 
            println("worker $worker launched job $new_job_id.")
            flush(stdout)
          end

        # Failed to launch new job
        else
          println("worker $worker failed to launch job $new_job_id.")
          flush(stdout)
          enqueue!(pending_jobs_id, new_job_id)
        end

      # Worker is (or was) busy
      elseif job_id > 0

        # Get status
        job_status = running_jobs[worker].state
        
        # Worker is done
        if job_status == :done
          
          # Fetch and reduce
          K .= redop(K, fetch(running_jobs[worker]))
          push!(done_jobs_id, job_id)
          if verbose
            println("worker $worker completed job $job_id.")
            flush(stdout)
          end
          
          # Free worker
          running_jobs_id[worker] = 0

        # Worker failed at completing its job
        elseif job_status == :failed
          println("worker $worker failed to complete job $job_id.")
          flush(stdout)
          enqueue!(pending_jobs_id, job_id)
          
          # Free worker
          running_jobs_id[worker] = 0
        
        # Job still running
        elseif job_status in (:running, :runnable)
          nothing

        # Status non-treated yet
        else
          println("worker $worker stopped running $job_id with status $(running_jobs[worker].state).")
          flush(stdout)
        end        
      end

    end # for (worker, job_id)
  end # while length(done_jobs_id) < njobs

end
  
#n = 2_000
#K = zeros(n, n)
#K = dynamic_mapreduce!(ones, + , [n for _ in 1:20], K)