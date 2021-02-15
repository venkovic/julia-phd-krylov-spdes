"""
     save_deflated_system(A::SparseMatrixCSC{Float64,Int},
                          b::Array{Float64,1},
                          W::Array{Float64,2},
                          s::Int,
                          precond_id::String;
                          print_error=false)

Saves linear system and basis of deflation subspace to JLD files.

"""
function save_deflated_system(A::SparseMatrixCSC{Float64,Int},
                              b::Array{Float64,1},
                              W::Array{Float64,2},
                              s::Int,
                              precond_id::String;
                              print_error=false)
  
  I, J, V = findnz(A)
  
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

"""
     save_system(A::SparseMatrixCSC{Float64,Int},
                 b::Array{Float64,1})

Saves linear system to JLD files.

"""
function save_system(A::SparseMatrixCSC{Float64,Int},
                     b::Array{Float64,1})

  I, J, V = findnz(A)

  JLD.save("data/A_I.jld", "I", I)
  JLD.save("data/A_J.jld", "J", J)
  JLD.save("data/A_V.jld", "V", V)

  JLD.save("data/b.jld", "b", b)
end

"""
     load_system()

Loads sparse linear system from JLD files.

"""
function load_system()

  I = JLD.load("data/A_I.jld", "I")
  J = JLD.load("data/A_J.jld", "J")
  V = JLD.load("data/A_V.jld", "V")
  A = sparse(I, J, V)

  b = JLD.load("data/b.jld", "b")

  return A, b
end