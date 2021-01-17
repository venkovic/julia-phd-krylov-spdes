using DelimitedFiles
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using LinearMaps


"""
set_subdomains(cells::Array{Int,2}, 
               cell_neighbors::Array{Int,2}, 
               epart::Array{Int64,2}, 
               npart::Array{Int64,2})
  
Returns helper data structures for non-overlaping domain decomposition using the mesh 
partition defined by epart and npart. 
  
Input:
  
mesh: Instance of TriangleMesh.TriMesh.

nel = mesh.n_cell
epart[iel, 1]: subdomain idom ∈ [1, ndom] to which element iel ∈ [1, nel] belongs.

nnode = mesh.n_point
npart[inode, 1]: subdomain idom ∈ [1, ndom] to which node inode ∈ [1, nnode] belongs.

Output:

elemd[idom][:]: 1D array of all the elements contained in subdomain idom ∈ [1, ndom].

node_Γ[:]: 1D array of global indices of the nodes at the interface of the mesh
           partition.


           
           ? IS node_Id necessary ?


node_Id[idom]: 1D array of global indices of the nodes strictly inside subdomain 
               idom ∈ [1, ndom].

ind_Id[idom]: Conversion table (i.e. dictionary) from global to local indices of 
              nodes strictly inside subdomain idom ∈ [1, ndom].

nnode_Id[idom]: Number of nodes strictly inside subdomain idom ∈ [1, ndom].

node_owner[inode]:    -1 if node inode ∈ [1, nnode] in Γ is not Dirchlet.
                       0 if node inode is Dirichlet.
                    idom ∈ [1, ndom] if node inode in (Ω_idom ∖ Γ) is not Dirichlet.
  
# Examples
```jldoctest
julia>
using TriangleMesh
using NPZ
using Fem

poly = polygon_unitSquare()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

ndom = 400
epart, npart = mesh_partition(mesh, ndom)
elemd, node_Γ, node_Id, ind_Id, nnode_Id, node_owner = set_subdomains(mesh, epart, npart)

```
"""
function set_subdomains(cells::Array{Int,2},
                        cell_neighbors::Array{Int,2},
                        epart::Array{Int64,1},
                        npart::Array{Int64,1},
                        dirichlet_inds_g2l::Dict{Int,Int})

    _, nel = size(cells) # Number of elements
    nnode = maximum(cells) # Number of nodes
    ndom = maximum(epart) # Number of subdomains

    # Essential data structures
    ind_Id_g2l = [Dict{Int, Int}() for _ in 1:ndom] # Conversion table from global to local indices of 
                                                    # nodes strictly inside each subdomain
    ind_Γd_g2l = [Dict{Int, Int}() for _ in 1:ndom] # Conversion table from global to local indices of 
                                                    # nodes on the interface of each subdomain
    ind_Γ_g2l = Dict{Int, Int}() # Conversion table from global to local indices of 
                                 # nodes on the interface of the mesh partition
    ind_Γd_Γ2l = [Dict{Int, Int}() for _ in 1:ndom] # Conversion table from indices of nodes in Γ to 
                                                    # local indices of nodes in Γ_d for each subdomain    
    node_owner = zeros(Int, nnode) # Indicates each node's owner 

    # Extra data structures
    elemd = [Int[] for _ in 1:ndom] # Elements contained inside each subdomain 
    node_Γ = Int[] # Nodes at the interface of the mesh partition which are not Dirichlet
    node_Γ_cnt = Int[] # Number of subdomains owning each local node ∈ Γ
    node_Id = [Int[] for _ in 1:ndom] # Non-Dirichlet nodes strictly inside each subdomain
    nnode_Id = zeros(Int, ndom) # Number of non-Dirchlet nodes strictly inside each subdomain

    # Correct potential indexing error of TriangleMesh.jl
    bnd_tag, iel_max = extrema(cell_neighbors)
    if iel_max > nel
      cell_neighbors .-= 1
      bnd_tag = -1
    end

    # Loop over elements in mesh
    iel_cell, jel_cell = zeros(Int, 3), zeros(Int, 3)
    for iel in 1:nel
  
      # Get global nodes and subdomain of element
      iel_cell .= cells[:, iel]
      idom = epart[iel]
      push!(elemd[idom], iel)
  
      # Loop over segments of element iel
      for j in 1:3
        jel = cell_neighbors[j, iel]

        # If segment is not on boundary
        if jel != bnd_tag
          
          # If neighbor belongs to another subdomain
          jdom = epart[jel]
          if jdom != idom
            jel_cell .= cells[:, jel]
  
            # Store common nodes in nodes_Γ
            for node in iel_cell

              # Pick non-dirichlet node on subdomain interface
              if (node in jel_cell) & !haskey(dirichlet_inds_g2l, node)
                
                if !haskey(ind_Γd_g2l[idom], node)
                  ind_Γd_g2l[idom][node] = ind_Γd_g2l[idom].count + 1
                end

                if !(node in node_Γ)
                  push!(node_Γ, node)
                  push!(node_Γ_cnt, 0)
                  ind_Γ_g2l[node] = ind_Γ_g2l.count + 1
                  node_owner[node] = -1
                end
              end
            end # for node
          end
        end
      end # end for j
    end

    # Store non-Dirichlet nodes in (Ω_d ∖ Γ)
    for inode in 1:nnode
      if !(haskey(dirichlet_inds_g2l, inode)) & (node_owner[inode] != -1)
        idom = npart[inode]
        push!(node_Id[idom], inode)
        node_owner[inode] = idom
      end
    end

    # Indexing of non-Dirichlet nodes in (Ω_d ∖ Γ)
    for idom in 1:ndom
      nnode_Id[idom] = length(node_Id[idom])
      for i in 1:nnode_Id[idom]
        ind_Id_g2l[idom][node_Id[idom][i]] = i
      end
    end

    # Create indexing conversion table from Γ to Γ_d for each subdomain
    for idom in 1:ndom
      for (gnode, _) in ind_Γd_g2l[idom]
        lnode_in_Γ = ind_Γ_g2l[gnode]
        node_Γ_cnt[lnode_in_Γ] += 1
        ind_Γd_Γ2l[idom][lnode_in_Γ] = ind_Γd_Γ2l[idom].count + 1
      end
    end

    return ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l,
           node_owner, elemd, node_Γ, node_Γ_cnt, node_Id, nnode_Id
end


function prepare_global_schur(cells::Array{Int,2},
                              points::Array{Float64,2},
                              epart::Array{Int,1},
                              ind_Id_g2l::Array{Dict{Int,Int},1},
                              ind_Γ_g2l::Dict{Int,Int},
                              node_owner::Array{Int,1},
                              coeff::Union{Array{Float64,1},
                                           Function},
                              f::Function,
                              uexact::Function)

  ndom = length(ind_Id_g2l) # Number of subdomains
  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of nodes
  n_Id = [ind_Id_g2l[idom].count for idom in 1:ndom] # Number of non-Dirichlet nodes strictly inside each subdomain
  n_Γ = ind_Γ_g2l.count # Number of non-Dirichlet nodes on the interface of the mesh partition  
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3)

  # Indices (I, J) and data (V) for sparse Galerkin operator A_IId
  IId_I = [Int[] for _ in 1:ndom]
  IId_J = [Int[] for _ in 1:ndom]
  IId_V = [Float64[] for _ in 1:ndom]

  # Indices (I, J) and data (V) for sparse Galerkin operators A_IΓd
  IΓd_I = [Int[] for _ in 1:ndom]
  IΓd_J = [Int[] for _ in 1:ndom]
  IΓd_V = [Float64[] for _ in 1:ndom]

  # Indices (I, J) and data (V) for sparse Galerkin operators A_ΓΓd
  ΓΓd_I = [Int[] for _ in 1:ndom]
  ΓΓd_J = [Int[] for _ in 1:ndom]
  ΓΓd_V = [Float64[] for _ in 1:ndom]

  # Indices (I, J) and data (V) for sparse Galerkin operator A_ΓΓ
  ΓΓ_I, ΓΓ_J, ΓΓ_V =  Int[], Int[], Float64[] 

  # Right hand sides
  b_Id = [zeros(Float64, n_Id[idom]) for idom in 1:ndom]
  b_Γ = zeros(Float64, n_Γ)

  # Loop over elements
  for iel in 1:nel

    # Get subdomain of element
    idom = epart[iel]

    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    Δa = 0.
    for (j, jnode) in enumerate(cells[:, iel])
      x[j], y[j] = points[1, jnode], points[2, jnode]
      if isa(coeff, Array{Float64,1})
        Δa += coeff[jnode]
      elseif isa(coeff, Function)
        Δa += coeff(x[j], y[j])
      end
    end
    Δa /= 3.

    # Terms of the shoelace formula for a triangle
    Δx[1] = x[3] - x[2]
    Δx[2] = x[1] - x[3]
    Δx[3] = x[2] - x[1]
    Δy[1] = y[2] - y[3]
    Δy[2] = y[3] - y[1]
    Δy[3] = y[1] - y[2]

    # Area of element
    Area = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.

    # Compute stiffness contributions
   for (i, inode) in enumerate(cells[:, iel])
     i_owner = node_owner[inode]
     i_is_dirichlet = i_owner == 0

      # Loop over vertices of element
      for (j, jnode) in enumerate(cells[:, iel])
        j_owner = node_owner[jnode]
        j_is_dirichlet = j_owner == 0

        # Evaluate contribution
        ΔKij = Δa * (Δy[i] * Δy[j] + Δx[i] * Δx[j]) / 4 / Area

        if !i_is_dirichlet & !j_is_dirichlet

          # inode, jnode ∈ Γ
          if (i_owner == -1) & (j_owner == -1)

            # Add contribution to A_ΓΓ:
            push!(ΓΓ_I, ind_Γ_g2l[inode])
            push!(ΓΓ_J, ind_Γ_g2l[jnode])
            push!(ΓΓ_V, ΔKij)

          # inode, jnode ∈ (Ω_idom \ Γ)
          elseif (i_owner > 0) & (j_owner > 0)

            # Add contribution to A_II:
            push!(IId_I[idom], ind_Id_g2l[idom][inode])
            push!(IId_J[idom], ind_Id_g2l[idom][jnode])
            push!(IId_V[idom], ΔKij)

          # inode ∉ Γ, jnode ∈ Γ
          elseif (i_owner > 0) & (j_owner == -1)

            # Add contribution to A_IΓ:
            push!(IΓd_I[idom], ind_Id_g2l[idom][inode])
            push!(IΓd_J[idom], ind_Γ_g2l[jnode])
            push!(IΓd_V[idom], ΔKij)
          end

        elseif i_is_dirichlet & !j_is_dirichlet

          # jnode ∈ Γ
          if j_owner == -1
            b_Γ[ind_Γ_g2l[jnode]] -= ΔKij * uexact(x[i], y[i])

          # jnode ∈ (Ω_idom \ Γ)
          elseif j_owner > 0
            b_Id[idom][ind_Id_g2l[idom][jnode]] -= ΔKij * uexact(x[i], y[i])
          end
        end # if
      end # for (j, jnode)
    end # for (i, inode)

    # Compute right hand side contributions
    for (i, inode) in enumerate(cells[:, iel])
      i_owner = node_owner[inode]
      i_is_dirichlet = i_owner == 0

      j = i + 1 - floor(Int, (i + 1) / 3) * 3
      j == 0 ? j = 3 : nothing
      k = i + 2 - floor(Int, (i + 2) / 3) * 3
      k == 0 ? k = 3 : nothing

      Δb = (2 * f(x[i], y[i]) 
              + f(x[j], y[j])
              + f(x[k], y[k])) * Area / 12

      # inode ∈ Γ
      if !i_is_dirichlet & (i_owner == -1)
        b_Γ[ind_Γ_g2l[inode]] += Δb

      # inode ∈ (Ω_idom \ Γ)
      elseif !i_is_dirichlet & (i_owner > 0)
        b_Id[idom][ind_Id_g2l[idom][inode]] += Δb
      end

    end # for (i, inode)
  end # for iel

  # Assemble csc sparse arrays
  A_IId = [sparse(IId_I[idom], IId_J[idom], IId_V[idom]) for idom in 1:ndom]
  A_IΓd = [sparse(IΓd_I[idom], IΓd_J[idom], IΓd_V[idom], n_Id[idom], n_Γ) for idom in 1:ndom]
  A_ΓΓ = sparse(ΓΓ_I, ΓΓ_J, ΓΓ_V)

  return A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ
end


function prepare_local_schurs(cells::Array{Int,2},
                              points::Array{Float64,2},
                              epart::Array{Int,1},
                              ind_Id_g2l::Array{Dict{Int,Int},1},
                              ind_Γd_g2l::Array{Dict{Int,Int},1},
                              ind_Γ_g2l::Dict{Int,Int},
                              node_owner::Array{Int,1},
                              coeff::Union{Array{Float64,1},
                                           Function},
                              f::Function,
                              uexact::Function)

  ndom = length(ind_Id_g2l) # Number of subdomains
  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of nodes
  n_Id = [ind_Id_g2l[idom].count for idom in 1:ndom] # Number of non-Dirichlet nodes strictly inside each subdomain
  n_Γd = [ind_Γd_g2l[idom].count for idom in 1:ndom] # Number of non-Dirichlet nodes on the interface of each subdomain
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3)

  # Indices (I, J) and data (V) for sparse Galerkin operator A_IId
  IId_I = [Int[] for _ in 1:ndom]
  IId_J = [Int[] for _ in 1:ndom]
  IId_V = [Float64[] for _ in 1:ndom]

  # Indices (I, J) and data (V) for sparse Galerkin operators A_IΓd
  IΓd_I = [Int[] for _ in 1:ndom]
  IΓd_J = [Int[] for _ in 1:ndom]
  IΓd_V = [Float64[] for _ in 1:ndom]

  # Indices (I, J) and data (V) for sparse Galerkin operators A_ΓΓd
  ΓΓd_I = [Int[] for _ in 1:ndom]
  ΓΓd_J = [Int[] for _ in 1:ndom]
  ΓΓd_V = [Float64[] for _ in 1:ndom]

  # Right hand sides
  b_Id = [zeros(Float64, n_Id[idom]) for idom in 1:ndom]
  b_Γ = zeros(Float64, sum(n_Γd))

  # Loop over elements
  for iel in 1:nel

    # Get subdomain of element
    idom = epart[iel]

    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    Δa = 0.
    for (j, jnode) in enumerate(cells[:, iel])
      x[j], y[j] = points[1, jnode], points[2, jnode]
      if isa(coeff, Array{Float64,1})
        Δa += coeff[jnode]
      elseif isa(coeff, Function)
        Δa += coeff(x[j], y[j])
      end
    end
    Δa /= 3.

    # Terms of the shoelace formula for a triangle
    Δx[1] = x[3] - x[2]
    Δx[2] = x[1] - x[3]
    Δx[3] = x[2] - x[1]
    Δy[1] = y[2] - y[3]
    Δy[2] = y[3] - y[1]
    Δy[3] = y[1] - y[2]

    # Area of element
    Area = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.

    # Compute stiffness contributions
    for (i, inode) in enumerate(cells[:, iel])
      i_owner = node_owner[inode]
      i_is_dirichlet = i_owner == 0

      # Loop over vertices of element
      for (j, jnode) in enumerate(cells[:, iel])
        j_owner = node_owner[jnode]
        j_is_dirichlet = j_owner == 0

        # Evaluate contribution
        ΔKij = Δa * (Δy[i] * Δy[j] + Δx[i] * Δx[j]) / 4 / Area

        if !i_is_dirichlet & !j_is_dirichlet

          # inode, jnode ∈ Γ
          if (i_owner == -1) & (j_owner == -1)

            # Add contribution to A_ΓΓd:
            push!(ΓΓd_I[idom], ind_Γd_g2l[idom][inode])
            push!(ΓΓd_J[idom], ind_Γd_g2l[idom][jnode])
            push!(ΓΓd_V[idom], ΔKij)

          # inode, jnode ∈ (Ω_idom \ Γ)
          elseif (i_owner > 0) & (j_owner > 0)

            # Add contribution to A_IId:
            push!(IId_I[idom], ind_Id_g2l[idom][inode])
            push!(IId_J[idom], ind_Id_g2l[idom][jnode])
            push!(IId_V[idom], ΔKij)

          # inode ∉ Γ, jnode ∈ Γ
          elseif (i_owner > 0) & (j_owner == -1)

            # Add contribution to A_IΓd:
            push!(IΓd_I[idom], ind_Id_g2l[idom][inode])
            push!(IΓd_J[idom], ind_Γd_g2l[idom][jnode])
            push!(IΓd_V[idom], ΔKij)
          end

        elseif i_is_dirichlet & !j_is_dirichlet

          # jnode ∈ Γ
          if j_owner == -1
            b_Γ[ind_Γ_g2l[jnode]] -= ΔKij * uexact(x[i], y[i])
  
          # jnode ∈ (Ω_idom \ Γ)
          elseif j_owner > 0
            b_Id[idom][ind_Id_g2l[idom][jnode]] -= ΔKij * uexact(x[i], y[i])
          end
        end # if
      end # for (j, jnode)
    end # for (i, inode)

    # Compute right hand side contributions
    for (i, inode) in enumerate(cells[:, iel])
      i_owner = node_owner[inode]
      i_is_dirichlet = i_owner == 0

      j = i + 1 - floor(Int, (i + 1) / 3) * 3
      j == 0 ? j = 3 : nothing
      k = i + 2 - floor(Int, (i + 2) / 3) * 3
      k == 0 ? k = 3 : nothing

      Δb = (2 * f(x[i], y[i]) 
              + f(x[j], y[j])
              + f(x[k], y[k])) * Area / 12

      # inode ∈ Γ
      if !i_is_dirichlet & (i_owner == -1)
        b_Γ[ind_Γ_g2l[inode]] += Δb

      # inode ∈ (Ω_idom \ Γ)
      elseif !i_is_dirichlet & (i_owner > 0)
        b_Id[idom][ind_Id_g2l[idom][inode]] += Δb
      end

    end # for (i, inode)
  end # for iel

  # Assemble csc sparse arrays
  A_IIdd = [sparse(IId_I[idom], IId_J[idom], IId_V[idom]) for idom in 1:ndom]
  A_IΓdd = [sparse(IΓd_I[idom], IΓd_J[idom], IΓd_V[idom], n_Id[idom], n_Γd[idom]) 
           for idom in 1:ndom]
  A_ΓΓdd = [sparse(ΓΓd_I[idom], ΓΓd_J[idom], ΓΓd_V[idom]) for idom in 1:ndom]

  return A_IIdd, A_IΓdd, A_ΓΓdd, b_Id, b_Γ
end


function apply_global_schur(A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                            A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                            A_ΓΓ::SparseMatrixCSC{Float64,Int},
                            x::Array{Float64,1};
                            preconds=nothing,
                            verbose=false)

  ndom = length(A_IId)
  Sx = A_ΓΓ * x
  for idom in 1:ndom
    if verbose
      print("cg solve $idom/$ndom ...")
      if isnothing(preconds)
        v = @time IterativeSolvers.cg(A_IId[idom], A_IΓd[idom] * x)
      else
        v = @time IterativeSolvers.cg(A_IId[idom], A_IΓd[idom] * x, Pl=preconds[idom])
      end
      print("SpMV $idom/$ndom ...")
      Sx .-= @time A_IΓd[idom]' * v
    else
      if isnothing(preconds)
        v = IterativeSolvers.cg(A_IId[idom], A_IΓd[idom] * x)
      else
        v = IterativeSolvers.cg(A_IId[idom], A_IΓd[idom] * x, Pl=preconds[idom])
      end
      Sx .-= A_IΓd[idom]' * v
    end
  end  
  return Sx
end


function apply_local_schur(A_IIdd::SparseMatrixCSC{Float64,Int},
                           A_IΓdd::SparseMatrixCSC{Float64,Int},
                           A_ΓΓdd::SparseMatrixCSC{Float64,Int},
                           xd::Array{Float64,1};
                           precond=nothing)
  
  Sdxd = A_ΓΓdd * xd
  if isnothing(precond)
    v = IterativeSolvers.cg(A_IIdd, A_IΓdd * xd, reltol=1e-9)
  else
    v = IterativeSolvers.cg(A_IIdd, A_IΓdd * xd, Pl=precond, reltol=1e-9)
  end
  Sdxd .-= A_IΓdd' * v
  return Sdxd
end


function apply_local_schurs(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                            A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                            A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                            ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                            node_Γ_cnt::Array{Int,1},
                            x::Array{Float64,1};
                            preconds=nothing)
  
  ndom = length(A_IIdd)
  Sx = zeros(Float64, length(node_Γ_cnt))

  for idom in 1:ndom

    xd = Array{Float64,1}(undef, ind_Γd_Γ2l[idom].count)
    Sdxd = Array{Float64,1}(undef, ind_Γd_Γ2l[idom].count)

    for (lnode_in_Γ, lnode_in_Γd) in ind_Γd_Γ2l[idom]
      xd[lnode_in_Γd] = x[lnode_in_Γ]
    end

    if isnothing(preconds)
      Sdxd .= apply_local_schur(A_IIdd[idom], A_IΓdd[idom], A_ΓΓdd[idom], xd)
    else
      Sdxd .= apply_local_schur(A_IIdd[idom], A_IΓdd[idom], A_ΓΓdd[idom], xd,
                                precond=preconds[idom])
    end

    for (lnode_in_Γ, lnode_in_Γd) in ind_Γd_Γ2l[idom]
      Sx[lnode_in_Γ] += Sdxd[lnode_in_Γd]
    end

  end # for idom

  return Sx
end


struct NeumannNeumannSchurPreconditioner
  ΠSd::Array{Cholesky{Float64,Array{Float64,2}},1}
  ind_Γd_Γ2l::Array{Dict{Int,Int},1}
  node_Γ_cnt::Array{Int,1}
end


function prepare_neumann_neumann_schur_precond(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                                               A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                               A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                               ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                                               node_Γ_cnt::Array{Int,1};
                                               preconds=nothing)

  ndom = length(A_IIdd)
  ΠSd = Cholesky{Float64,Array{Float64,2}}[]

  for idom in 1:ndom
    if isnothing(preconds)
      Sd = LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                             A_IΓdd[idom],
                                             A_ΓΓdd[idom],
                                             xd),
                                             ind_Γd_Γ2l[idom].count, issymmetric=true)
    else
      Sd = LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                             A_IΓdd[idom],
                                             A_ΓΓdd[idom],
                                             xd,
                                             precond=preconds[idom]),
                                             ind_Γd_Γ2l[idom].count, issymmetric=true)
    end

    push!(ΠSd, cholesky(Symmetric(sparse(Sd))))
  end

  return NeumannNeumannSchurPreconditioner(ΠSd,
                                           ind_Γd_Γ2l,
                                           node_Γ_cnt)
end


function apply_neumann_neumann(Πnn::NeumannNeumannSchurPreconditioner,
                               r::Array{Float64,1})

  ndom = length(Πnn.ΠSd)
  z = zeros(Float64, length(Πnn.node_Γ_cnt))

  for idom in 1:ndom

    n_Γd = Πnn.ind_Γd_Γ2l[idom].count
    rd = Array{Float64,1}(undef, n_Γd)
    ΠSdrd = Array{Float64,1}(undef, n_Γd)

    for (lnode_in_Γ, lnode_in_Γd) in Πnn.ind_Γd_Γ2l[idom]
      rd[lnode_in_Γd] = r[lnode_in_Γ] / Πnn.node_Γ_cnt[lnode_in_Γ]
    end

    ΠSdrd .= Πnn.ΠSd[idom] \ rd

    for (lnode_in_Γ, lnode_in_Γd) in Πnn.ind_Γd_Γ2l[idom]
      z[lnode_in_Γ] += ΠSdrd[lnode_in_Γd] / node_Γ_cnt[lnode_in_Γ]
    end

   end # for idom

   return z
end


import Base: \
function (\)(Πnn::NeumannNeumannSchurPreconditioner, x::Array{Float64,1})
  apply_neumann_neumann(Πnn, x)
end


function LinearAlgebra.ldiv!(z::Array{Float64,1}, 
                             Πnn::NeumannNeumannSchurPreconditioner,
                             r::Array{Float64,1})

  z .= apply_neumann_neumann(Πnn, r)
end


function LinearAlgebra.ldiv!(Πnn::NeumannNeumannSchurPreconditioner,
                             r::Array{Float64,1})

r .= apply_neumann_neumann(Πnn, copy(r))
end


function get_schur_rhs(b_Id::Array{Array{Float64,1},1},
                       A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                       A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                       b_Γ::Array{Float64,1})
  ndom = length(b_Id)
  n_Γ = length(b_Γ)
  n_Id = [length(b_Id[idom]) for idom in 1:ndom]

  b_schur = Array{Float64,1}(undef, length(b_Γ))
  b_schur .= b_Γ

  for idom in 1:ndom
    v = IterativeSolvers.cg(A_IId[idom], b_Id[idom])
    b_schur .-= A_IΓd[idom]'v
  end

  return b_schur
end


function get_subdomain_solutions(u_Γ::Array{Float64,1},
                                 A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                                 A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                                 b_Id::Array{Array{Float64,1},1})
  ndom = length(b_Id)
  u_Id = [Array{Float64,1}(undef, length(b_Id[idom])) for idom in 1:ndom]
  for idom in 1:ndom
    u_Id[idom] .= IterativeSolvers.cg(A_IId[idom], b_Id[idom] .- A_IΓd[idom] * u_Γ)
  end

  return u_Id
end


function merge_subdomain_solutions(u_Γ::Array{Float64,1},
                                   u_Id::Array{Array{Float64,1},1},
                                   node_Γ::Array{Int,1},
                                   node_Id::Array{Array{Int,1},1},
                                   dirichlet_inds_l2g::Array{Int,1},
                                   uexact::Function,
                                   points::Array{Float64,2})
  
  ndom = length(u_Id)
  n_Γ = length(u_Γ)
  n_Id = [length(u_Id[idom]) for idom in 1:ndom]
  u = Array{Float64,1}(undef, n_Γ 
                            + sum(n_Id) 
                            + length(dirichlet_inds_l2g))

  for (i, inode) in enumerate(node_Γ)
    u[inode] = u_Γ[i]
  end

  for idom in 1:ndom
    for (i, inode) in enumerate(node_Id[idom])
      u[inode] = u_Id[idom][i]
    end
  end

  for (i, inode) in enumerate(dirichlet_inds_l2g)
    u[inode] = uexact(points[1, inode], points[2, inode])
  end

  return u
end


function assemble_A_ΓΓ_from_local_blocks(A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
  ind_Γd_Γ2l::Array{Dict{Int,Int},1})

ndom = length(ind_Γd_Γ2l)
ΓΓ_I = Int[]
ΓΓ_J = Int[]
ΓΓ_V = Float64[]

for idom in 1:ndom

n_Γd = length(ind_Γd_Γ2l[idom])
ind_Γd_l2Γ = Array{Float64,1}(undef, n_Γd)

for (lnode_in_Γ, lnode_in_Γd) in ind_Γd_Γ2l[idom]
ind_Γd_l2Γ[lnode_in_Γd] = lnode_in_Γ
end

jnode = 1
for (i, inode) in enumerate(A_ΓΓdd[idom].rowval)
push!(ΓΓ_I, ind_Γd_l2Γ[inode])
push!(ΓΓ_J, ind_Γd_l2Γ[jnode])
push!(ΓΓ_V, A_ΓΓdd[idom][inode, jnode])
i == A_ΓΓdd[idom].colptr[jnode + 1] ? jnode += 1 : nothing
end
end
A_ΓΓ = sparse(ΓΓ_I, ΓΓ_J, ΓΓ_V)
return A_ΓΓ
end