function save_deflated_system(A::SparseMatrixCSC{Float64,Int},
                              b::Array{Float64,1},
                              W::Array{Float64,2},
                              s::Int,
                              precond_id::String;
                              print_error=false)
  
  I, J, V = sparse(A)
  
  JLD.save("data/A_$(s)_I.jld", "I", I)
  JLD.save("data/A_$(s)_J.jld", "J", J)
  JLD.save("data/A_$(s)_V.jld", "V", V)
  
  JLD.save("data/b_$(s).jld", "b", b)
  
  JLD.save("data/W_$(precond_id)_$(s).jld", "W", W)
  
  if print_error
    println("Warning: encountered a problem with eigdefpcg,
             data was saved for further investigation")
  end
end