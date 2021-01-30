"""
     function suggest_parameters(nnode::Int)
  
Returns suggested values for relative_local and relative_global, 
to be used for the truncation of local and global Karhunen expansions.

(relative_local, relative_global) = (.9993, .995), for a SExp 
covariance model with sig2=1 and L=0.1, leads to satisfactory results 
in these cases:

  ndom = 400, dev = 25, tentative_nnode =   200_000, forget=1e-6

  ndom = 400, dev = 25, tentative_nnode =   400_000, forget=1e-6
  
  ndom = 500, dev = 35, tentative_nnode = 1_000_000, forget=1e-6

"""
function suggest_parameters(nnode::Int)
  return .9993, .995 
end


"""
     function get_root_filename(model::String,
                                sig2::Float64,
                                L::Float64,
                                nnode::Int)

Returns filenames' basis for covariance model and # of DoFs.

"""
function get_root_filename(model::String,
                           sig2::Float64,
                           L::Float64,
                           nnode::Int)
  fname = model * "_"
  fname *= "sig2$sig2" * "_"
  fname *= "L$L" * "_"
  return fname * "DoF$(nnode)"
end