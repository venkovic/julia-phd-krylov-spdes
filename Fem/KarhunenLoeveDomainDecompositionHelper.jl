function suggest_parameters(nnode::Int)
  # Works well for SExp, sig2=1, L=0.1 with
  #  ndom = 400, dev = 25, tentative_nnode = 200_000, forget=1e-6
  #  ndom = 400, dev = 25, tentative_nnode = 400_000, forget=1e-6
  return .9993, .995 
end


function get_root_filename(model::String,
                           sig2::Float64,
                           L::Float64,
                           nnode::Int)
  fname = model * "_"
  fname *= "sig2$sig2" * "_"
  fname *= "L$L" * "_"
  return fname * "DoF$(nnode)"
end

