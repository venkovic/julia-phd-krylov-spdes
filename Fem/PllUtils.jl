"""
     function dynamic_map!(func::Function,
                           coll::Union{UnitRange{Int},  
                                       Array{Int,1}},
                           domains::Dict{Int,SubDomain};
                           verbose=true,
                           Δt=2.)

Does parallel map from collection `coll` of subdomains with the 
function `func`, doing a dynamic task scheduling over multiple hosts.

Input:

 `func::Function`,
  function used to map the collection of subdomains.

 `coll::Union{UnitRange{Int},
              Array{Int,1}}`.

 `domains::Dict{Int,SubDomain}`,
  dictionary of subdomains.

 `verbose=true`.

 `Δt=2.`,
  time elapsed between checks on workers.

"""
function dynamic_map!(func::Function,
                      coll::Union{UnitRange{Int},  
                                  Array{Int,1}},
                      domains::Dict{Int,SubDomain};
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
          domains[job_id] = fetch(running_jobs[worker])
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
          cnt_failures[worker] += 1

          # Terminate worker
          if cnt_failures[worker] >= nfails_allowed 
            running_jobs_id[worker] = -1
          end     
        end

      end # if job_id ...
    end # for (worker, job_id)
  end # while length(done_jobs_id) < njobs
end
