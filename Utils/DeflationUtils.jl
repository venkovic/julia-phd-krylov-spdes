"""
     save_deflated_system(A::SparseMatrixCSC{Float64,Int},
                          b::Array{Float64,1},
                          W::Array{Float64,2};
                          s=0,
                          precond_id="",
                          print_error=false)

Saves linear system and basis of deflation subspace to JLD files.

"""
function save_deflated_system(A::SparseMatrixCSC{Float64,Int},
                              b::Array{Float64,1},
                              W::Array{Float64,2};
                              s=0,
                              precond_id="",
                              print_error=false)
  
  I, J, V = findnz(A)
  
  if s == 0
    JLD.save("data/A_I.jld", "I", I)
    JLD.save("data/A_J.jld", "J", J)
    JLD.save("data/A_V.jld", "V", V)
    JLD.save("data/b.jld", "b", b)
  else
    JLD.save("data/A_$(s)_I.jld", "I", I)
    JLD.save("data/A_$(s)_J.jld", "J", J)
    JLD.save("data/A_$(s)_V.jld", "V", V)
    JLD.save("data/b_$(s).jld", "b", b)
  end
   
  if precond_id == "" && s == 0
    JLD.save("data/W.jld", "W", W)
  else
    JLD.save("data/W_$(precond_id)_$(s).jld", "W", W)
  end

  if print_error
    println("Warning: encountered a problem with eigdefpcg,
             data was saved for further investigation")
  end
end


"""
     load_deflated_system(A::SparseMatrixCSC{Float64,Int},
                          b::Array{Float64,1},
                          W::Array{Float64,2},
                          s::Int,
                          precond_id::String;
                          print_error=false)

Loads linear system and basis of deflation subspace from JLD files.

"""
function load_deflated_system(s::Int,
                              precond_id::String)
  
  I = JLD.load("data/A_$(s)_I.jld", "I")
  J = JLD.load("data/A_$(s)_J.jld", "J")
  V = JLD.load("data/A_$(s)_V.jld", "V")
  A = sparse(I, J, V)
  
  b = JLD.load("data/b_$(s).jld", "b")
  
  W = JLD.load("data/W_$(precond_id)_$(s).jld", "W")

  return A, b, W
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