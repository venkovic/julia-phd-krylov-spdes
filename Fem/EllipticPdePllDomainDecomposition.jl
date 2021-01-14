


# Distributed computation of local KL expansion for each subdomain  
@time domain = @sync @distributed merge! for idom in 1:ndom
  relative_local, _ = suggest_parameters(mesh.n_point)
  pll_solve_local_kl(mesh, epart, cov, nev, idom, 
                     relative=relative_local)
end




function pll_apply_schur(A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                         A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                         A_ΓΓ::SparseMatrixCSC{Float64,Int},
                         x::Array{Float64,1},
                         Π_IId;
                         verbose=true)

ndom = length(A_IId)

Sx = @sync @distributed (+) for idom in 1:ndom
  v = IterativeSolvers.cg(A_IId[idom], A_IΓd[idom] * x, Pl=Π_IId[idom])
  A_IΓd[idom]' * v
end
Sx .+= A_ΓΓ * x


return Sx
end