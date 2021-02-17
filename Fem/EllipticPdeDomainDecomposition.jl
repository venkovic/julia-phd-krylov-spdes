"""
     set_subdomains(cells::Array{Int,2},
                    cell_neighbors::Array{Int,2},
                    epart::Array{Int64,1},
                    npart::Array{Int64,1},
                    dirichlet_inds_g2l::Dict{Int,Int})
  
Returns helper data structures for non-overlaping domain decomposition (edge-based
partitioning) using the partition defined by (`epart`, `npart`), for the mesh with
(`cells`, `cell_neighbors`) and non-Dirichlet nodes in `dirichlet_inds_g2l`.
  
Input:

 `cells::Array{Int,2}`, `size(cells) = (3, n_el)`,
  nodes of each element.

 `cell_neighbors::Array{Int,2}`, `size(cell_neighbors) = (3, n_el)`,
  neighboring elements of each element, or `-1` if edge is at mesh boundary. 

 `epart::Array{Int64,1}`, `size(epart) = (n_el,)`,
  host subdomain of each element.

 `npart::Array{Int,1}`, `size(npart) = (nnode,)`,
  a host subdomain of each node.

 `dirichlet_inds_g2l::Dict{Int,Int}`, `dirichlet_inds_g2l.count` = # non-Dirichlet nodes
  Conversion table from global mesh node indices to non-dirichlet indices.

Output:

 `ind_Id_g2l::Array{Dict{Int,Int}}`, 
  conversion tables from global to local indices of nodes strictly inside each subdomain.

 `ind_Γd_g2l::Array{Dict{Int,Int}}`,
  conversion table from global to local indices of nodes on the interface of each subdomain.

 `ind_Γ_g2l::Dict{Int,Int}`,
  conversion table from global to local indices of nodes on the interface of the mesh partition.

 `ind_Γd_Γ2l::Array{Dict{Int,Int}}`,
  conversion table from indices of nodes in Γ to local indices of nodes in Γ_d for each subdomain.

 `node_owner::Array{Int,1}`,
  indicates each node's owner: `-1` if node `inode` ∈ [`1`, `nnode`] in Γ is not Dirchlet,
                                `0` if node `inode` is Dirichlet,
             `idom` ∈ [`1`, `ndom`] if node `inode` in (Ω_idom ∖ Γ) is not Dirichlet.

 `elemd::Array{Array{Int,1},1}`
  Elements contained inside each subdomain, i.e., `elemd[idom][:]` is an array of all the elements 
  contained in subdomain `idom` ∈ [`1`, `ndom`].

 `node_Γ::Array{Int,1}`
  Nodes at the interface of the mesh partition which are not Dirichlet

 `node_Γ_cnt::Array{Int,1}`
  Number of subdomains owning each local node ∈ Γ

 `node_Id::Array{Array{Int,1},1}`
  Non-Dirichlet nodes strictly inside each subdomain

 `nnode_Id::Array{Int,1}`
  Number of non-Dirchlet nodes strictly inside each subdomain
  
# Examples
```jldoctest
julia>
using Fem

ndom = 400
tentative_nnode = 400_000
mesh = get_mesh(tentative_nnode)

dirichlet_inds_g2l, not_dirichlet_inds_g2l,
dirichlet_inds_l2g, not_dirichlet_inds_l2g = 
get_dirichlet_inds(mesh.point, mesh.point_marker)

ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l, node_owner,
elemd, node_Γ, node_Γ_cnt, node_Id, nnode_Id = set_subdomains(cells,
                                                              cell_neighbors,
                                                              epart, 
                                                              npart,
                                                              dirichlet_inds_g2l)

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
      for (gnode, lnode_in_Γd) in ind_Γd_g2l[idom]
        lnode_in_Γ = ind_Γ_g2l[gnode]
        node_Γ_cnt[lnode_in_Γ] += 1
        ind_Γd_Γ2l[idom][lnode_in_Γ] = lnode_in_Γd
      end
    end

    return ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l,
           node_owner, elemd, node_Γ, node_Γ_cnt, node_Id, nnode_Id
end


"""
     prepare_global_schur(cells::Array{Int,2},
                          points::Array{Float64,2},
                          epart::Array{Int,1},
                          ind_Id_g2l::Array{Dict{Int,Int},1},
                          ind_Γ_g2l::Dict{Int,Int},
                          node_owner::Array{Int,1},
                          coeff::Union{Array{Float64,1},
                                       Function},
                          f::Function,
                          uexact::Function)
  
Prepares and returns matrices necessary for the application 
and/or assembly of the global Schur complement. 
  
"""
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


"""
     prepare_local_schurs(cells::Array{Int,2},
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
  
Prepares and returns all matrices necessary for the application 
and/or assembly of local Schur complements. 
  
"""
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
  n_Γ = ind_Γ_g2l.count
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


"""
     apply_global_schur(A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                        A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                        A_ΓΓ::SparseMatrixCSC{Float64,Int},
                        x::Array{Float64,1};
                        preconds=nothing,
                        verbose=false)
  
Applies global Schur complement with local (p)cg solves. 
  
"""
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


"""
     apply_local_schur(A_IIdd::SparseMatrixCSC{Float64,Int},
                       A_IΓdd::SparseMatrixCSC{Float64,Int},
                       A_ΓΓdd::SparseMatrixCSC{Float64,Int},
                       xd::Array{Float64,1};
                       precond=nothing,
                       reltol=1e-9)
  
Applies a single local Schur complement with (p)cg solve. 
  
"""
function apply_local_schur(A_IIdd::SparseMatrixCSC{Float64,Int},
                           A_IΓdd::SparseMatrixCSC{Float64,Int},
                           A_ΓΓdd::SparseMatrixCSC{Float64,Int},
                           xd::Array{Float64,1};
                           precond=nothing,
                           reltol=1e-9)
  
  Sdxd = A_ΓΓdd * xd
  if isnothing(precond)
    v = IterativeSolvers.cg(A_IIdd, A_IΓdd * xd, reltol=reltol)
  else
    v = IterativeSolvers.cg(A_IIdd, A_IΓdd * xd, Pl=precond, reltol=reltol)
  end
  Sdxd .-= A_IΓdd' * v
  return Sdxd
end


"""
     assemble_local_schurs(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                           A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                           A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1};
                           preconds=nothing,
                           reltol=1e-9)
  
Assembles local Schur complements. 
  
"""
function assemble_local_schurs(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                               A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                               A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1};
                               preconds=nothing,
                               reltol=1e-9)

  Sd_sp = SparseMatrixCSC{Float64,Int}[]
  ndom = length(A_ΓΓdd)
  for idom = 1:ndom
    if isnothing(preconds)
      Sd_map = LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                                 A_IΓdd[idom],
                                                 A_ΓΓdd[idom],
                                                 xd,
                                                 precond=preconds[idom],
                                                 reltol=reltol),
                                                 A_ΓΓdd[idom].n, issymmetric=true)
    else
      Sd_map = LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                                 A_IΓdd[idom],
                                                 A_ΓΓdd[idom],
                                                 xd,
                                                 reltol=reltol),
                                                 A_ΓΓdd[idom].n, issymmetric=true)
    end
    push!(Sd_sp, sparse(Symmetric(Array(Sd_map))))
  end
return Sd_sp
end


"""
     apply_local_schurs(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                        A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                        A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                        ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                        node_Γ_cnt::Array{Int,1},
                        x::Array{Float64,1};
                        preconds=nothing,
                        reltol=1e-9)
  
Applies local Schur complements with (p)cg solves, and gather. 
  
"""
function apply_local_schurs(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                            A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                            A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                            ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                            node_Γ_cnt::Array{Int,1},
                            x::Array{Float64,1};
                            preconds=nothing,
                            reltol=1e-9)
  
  ndom = length(A_IIdd)
  Sx = zeros(Float64, length(node_Γ_cnt))

  for idom in 1:ndom

    xd = Array{Float64,1}(undef, ind_Γd_Γ2l[idom].count)
    Sdxd = Array{Float64,1}(undef, ind_Γd_Γ2l[idom].count)

    for (lnode_in_Γ, lnode_in_Γd) in ind_Γd_Γ2l[idom]
      xd[lnode_in_Γd] = x[lnode_in_Γ]
    end

    if isnothing(preconds)
      Sdxd .= apply_local_schur(A_IIdd[idom], A_IΓdd[idom], A_ΓΓdd[idom], xd,
                                reltol=reltol)
    else
      Sdxd .= apply_local_schur(A_IIdd[idom], A_IΓdd[idom], A_ΓΓdd[idom], xd,
                                precond=preconds[idom], reltol=reltol)
    end

    for (lnode_in_Γ, lnode_in_Γd) in ind_Γd_Γ2l[idom]
      Sx[lnode_in_Γ] += Sdxd[lnode_in_Γd]
    end

  end # for idom

  return Sx
end


"""
     apply_local_schurs(Sd::Union{Array{SparseMatrixCSC{Float64,Int},1},
                                  Array{SparseMatrixCSC{Float64,Int},1}},
                        ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                        node_Γ_cnt::Array{Int,1},
                        x::Array{Float64,1};
                        reltol=1e-9)
  
Applies (previously assembled) local Schur complements, and gather. 
  
"""
function apply_local_schurs(Sd::Union{Array{SparseMatrixCSC{Float64,Int},1},
                                      Array{SparseMatrixCSC{Float64,Int},1}},
                            ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                            node_Γ_cnt::Array{Int,1},
                            x::Array{Float64,1};
                            reltol=1e-9)

  ndom = length(Sd)
  Sx = zeros(Float64, length(node_Γ_cnt))

  for idom in 1:ndom
    xd = Array{Float64,1}(undef, ind_Γd_Γ2l[idom].count)
    Sdxd = Array{Float64,1}(undef, ind_Γd_Γ2l[idom].count)

    for (lnode_in_Γ, lnode_in_Γd) in ind_Γd_Γ2l[idom]
      xd[lnode_in_Γd] = x[lnode_in_Γ]
    end
    Sdxd .= Sd[idom] * xd
    for (lnode_in_Γ, lnode_in_Γd) in ind_Γd_Γ2l[idom]
      Sx[lnode_in_Γ] += Sdxd[lnode_in_Γd]
    end
  end # for idom

  return Sx
end


"""
     get_schur_rhs(b_Id::Array{Array{Float64,1},1},
                   A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                   A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                   b_Γ::Array{Float64,1};
                   preconds=nothing)
  
Assembles the Schur right hand side.
  
"""
function get_schur_rhs(b_Id::Array{Array{Float64,1},1},
                       A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                       A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                       b_Γ::Array{Float64,1};
                       preconds=nothing)
  ndom = length(b_Id)
  n_Γ = length(b_Γ)
  n_Id = [length(b_Id[idom]) for idom in 1:ndom]

  b_schur = Array{Float64,1}(undef, length(b_Γ))
  b_schur .= b_Γ

  for idom in 1:ndom
    v = Array{Float64,1}(undef, A_IId[idom].n)
    if isnothing(preconds)
      v .= IterativeSolvers.cg(A_IId[idom], b_Id[idom])
    else
      v .= IterativeSolvers.cg(A_IId[idom], b_Id[idom], Pl=preconds[idom])
    end
    b_schur .-= A_IΓd[idom]'v
  end

  return b_schur
end


"""
     get_schur_rhs(b_Id::Array{Array{Float64,1},1},
                   A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                   A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                   b_Γ::Array{Float64,1},
                   ind_Γd_Γ2l::Array{Dict{Int,Int}};
                   preconds=nothing)
  
Assembles the Schur right hand side.
  
"""
function get_schur_rhs(b_Id::Array{Array{Float64,1},1},
                       A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                       A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                       b_Γ::Array{Float64,1},
                       ind_Γd_Γ2l::Array{Dict{Int,Int}};
                       preconds=nothing)

  ndom = length(b_Id)
  n_Γ = length(b_Γ)
  n_Id = [length(b_Id[idom]) for idom in 1:ndom]

  b_schur = Array{Float64,1}(undef, length(b_Γ))
  b_schur .= b_Γ

  for idom in 1:ndom
    v = Array{Float64,1}(undef, A_IId[idom].n)
    w = Array{Float64,1}(undef, A_IΓd[idom].n)
    if isnothing(preconds)
      v .= IterativeSolvers.cg(A_IId[idom], b_Id[idom])
    else
      v .= IterativeSolvers.cg(A_IId[idom], b_Id[idom], Pl=preconds[idom])
    end
    w .= A_IΓd[idom]'v
    for (lnode_in_Γ, lnode_in_Γd) in ind_Γd_Γ2l[idom]
      b_schur[lnode_in_Γ] -= w[lnode_in_Γd]
    end
  end

  return b_schur
end


"""
     get_subdomain_solutions(u_Γ::Array{Float64,1},
                             A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                             A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                             b_Id::Array{Array{Float64,1},1})
  
Computes `u_I`.
  
"""
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


"""
     merge_subdomain_solutions(u_Γ::Array{Float64,1},
                               u_Id::Array{Array{Float64,1},1},
                               node_Γ::Array{Int,1},
                               node_Id::Array{Array{Int,1},1},
                               dirichlet_inds_l2g::Array{Int,1},
                               uexact::Function,
                               points::Array{Float64,2})
  
Assembles solution with dirichlet bcs.
  
"""
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


"""
     assemble_A_ΓΓ_from_local_blocks(A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                     ind_Γd_Γ2l::Array{Dict{Int,Int},1})
  
Assembles `A_ΓΓ` from `A_ΓΓd`'s.
  
"""
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
      i + 1 == A_ΓΓdd[idom].colptr[jnode + 1] ? jnode += 1 : nothing
    end
  end

  A_ΓΓ = sparse(ΓΓ_I, ΓΓ_J, ΓΓ_V)
  return A_ΓΓ
end


struct NeumannNeumannSchurPreconditioner

  # Le Tallec P, De Roeck YH, Vidrascu M.
  # Domain decomposition methods for large linearly elliptic three-dimensional problems. 
  # Journal of Computational and Applied Mathematics. 1991 Feb 10;34(1):93-117.

  # Bourgat JF, Glowinski R, Le Tallec P, Vidrascu M. 
  # Variational formulation and algorithm for trace operator in domain decomposition calculations. 
  # Inria research report. 1988;RR-804:pp.18.

  # Giraud L, Tuminaro RS. 
  # Algebraic domain decomposition preconditioners. 
  # Mesh partitioning techniques and domain decomposition methods. 2006 Oct:187-216.

  ΠSd::Array{Array{Float64,2},1}
  ind_Γd_Γ2l::Array{Dict{Int,Int},1}
  node_Γ_cnt::Array{Int,1}
end


"""
     prepare_neumann_neumann_schur_precond(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                                           A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                           A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                           ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                                           node_Γ_cnt::Array{Int,1};
                                           preconds=nothing)
  
Prepares and returns a `NeumannNeumannSchurPreconditioner`.
Computes pseudo-inverses of (singular) local Schur complements.
  
"""
function prepare_neumann_neumann_schur_precond(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                                               A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                               A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                               ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                                               node_Γ_cnt::Array{Int,1};
                                               preconds=nothing)

  ndom = length(A_IIdd)
  ΠSd = Array{Float64,2}[]

  for idom in 1:ndom
    if isnothing(preconds)
      Sd = LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                             A_IΓdd[idom],
                                             A_ΓΓdd[idom],
                                             xd,
                                             reltol=1e-15),
                                             ind_Γd_Γ2l[idom].count, issymmetric=true)
    else
      Sd = LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                             A_IΓdd[idom],
                                             A_ΓΓdd[idom],
                                             xd,
                                             precond=preconds[idom],
                                             reltol=1e-15),
                                             ind_Γd_Γ2l[idom].count, issymmetric=true)
    end

    Sd_mat = Array(Sd)
    pinv_Sd = LinearAlgebra.pinv(Sd_mat, rtol=sqrt(eps(real(float(one(eltype(Sd_mat)))))))

    push!(ΠSd, pinv_Sd)
  end

  return NeumannNeumannSchurPreconditioner(ΠSd,
                                           ind_Γd_Γ2l,
                                           node_Γ_cnt)
end


"""
     prepare_neumann_neumann_schur_precond(Sd_local_mat::Array{SparseMatrixCSC{Float64,Int},1},
                                           ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                                           node_Γ_cnt::Array{Int,1})
  
Prepares and returns a `NeumannNeumannSchurPreconditioner`.
Computes pseudo-inverses of (singular) local Schur complements.
  
"""
function prepare_neumann_neumann_schur_precond(Sd_local_mat::Array{SparseMatrixCSC{Float64,Int},1},
                                               ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                                               node_Γ_cnt::Array{Int,1})

  ndom = length(Sd_local_mat)
  ΠSd = Array{Float64,2}[]

  for idom in 1:ndom
    rtol = sqrt(eps(real(float(one(eltype(Sd_local_mat[idom]))))))
    if isa(Sd_local_mat[idom], SparseMatrixCSC)
      push!(ΠSd, LinearAlgebra.pinv(Array(Sd_local_mat[idom]), rtol=rtol))
    else
      push!(ΠSd, LinearAlgebra.pinv(Sd_local_mat[idom], rtol=rtol))
    end
  end

  return NeumannNeumannSchurPreconditioner(ΠSd,
                                           ind_Γd_Γ2l,
                                           node_Γ_cnt)
end


"""
     apply_neumann_neumann_schur(Πnn::NeumannNeumannSchurPreconditioner,
                                r::Array{Float64,1})
  
Applies the `NeumannNeumannSchurPreconditioner`.
  
"""
function apply_neumann_neumann_schur(Πnn::NeumannNeumannSchurPreconditioner,
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

    ΠSdrd .= Πnn.ΠSd[idom] * rd

    for (lnode_in_Γ, lnode_in_Γd) in Πnn.ind_Γd_Γ2l[idom]
      z[lnode_in_Γ] += ΠSdrd[lnode_in_Γd] / Πnn.node_Γ_cnt[lnode_in_Γ]
    end

   end # for idom

   return z
end


import Base: \
function (\)(Πnn::NeumannNeumannSchurPreconditioner, x::Array{Float64,1})
  apply_neumann_neumann_schur(Πnn, x)
end

function LinearAlgebra.ldiv!(z::Array{Float64,1}, 
                             Πnn::NeumannNeumannSchurPreconditioner,
                             r::Array{Float64,1})
  z .= apply_neumann_neumann_schur(Πnn, r)
end

function LinearAlgebra.ldiv!(Πnn::NeumannNeumannSchurPreconditioner,
                             r::Array{Float64,1})
  r .= apply_neumann_neumann_schur(Πnn, copy(r))
end


struct LorascPreconditioner

  # Grigori L, Frédéric N, Soleiman Y.
  # Robust algebraic Schur complement preconditioners based on low rank corrections.
  # Inria research report. 2014;RR-8557:pp.18.

  A_IΓd::Array{SparseMatrixCSC{Float64,Int},1}
  chol_A_IId::Array{SuiteSparse.CHOLMOD.Factor{Float64},1}
  A_ΓΓ::SparseMatrixCSC{Float64,Int}
  ΠA_ΓΓ # preconditioner for A_ΓΓ
  chol_A_ΓΓ::Union{SuiteSparse.CHOLMOD.Factor{Float64},Nothing}
  ε::Float64
  E::Array{Array{Float64,1},1}
  Σ::Array{Float64,1}
  ind_Id_g2l::Array{Dict{Int,Int},1}
  ind_Γ_g2l::Dict{Int,Int}
  not_dirichlet_inds_g2l::Dict{Int,Int}

  x_Id::Array{Array{Float64,1},1}
  x_Γ::Array{Float64, 1}
  z_Γ::Array{Float64, 1}
  u::Array{Float64,1}
end


"""
     apply_bmat!(A_ΓΓ::SparseMatrixCSC{Float64,Int},
                 chol_A_ΓΓ::SuiteSparse.CHOLMOD.Factor{Float64},
                 S::FunctionMap,
                 x::Array{Float64,1})

Computes L  ((A_ΓΓ - S) * (L'  x)), where A_ΓΓ = L*L', 
as needed for the randomized computation of low rank corrections.

"""
function apply_bmat!(A_ΓΓ::SparseMatrixCSC{Float64,Int},
                     chol_A_ΓΓ::SuiteSparse.CHOLMOD.Factor{Float64},
                     S::FunctionMap,
                     x::Array{Float64,1})

  y = Array{Float64,1}(undef, A_ΓΓ.n)
  y .= chol_A_ΓΓ.L' \ x
  x .= S * y
  y .= A_ΓΓ * y .- x
  x .= chol_A_ΓΓ.L \ y
end


"""
     prepare_lorasc_precond(S::FunctionMap{Float64},
                            A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                            A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                            A_ΓΓ::SparseMatrixCSC{Float64,Int},
                            ind_Id_g2l::Array{Dict{Int,Int},1},
                            ind_Γ_g2l::Dict{Int,Int}, 
                            not_dirichlet_inds_g2l::Dict{Int,Int};
                            verbose=true,
                            compute_A_ΓΓ_chol=true,
                            nvec=25, 
                            ε=.01)

Prepares and returns a `LorascPreconditioner`.
See Grigori et al. (2014) for a reference on the LORASC preconditioner.

Grigori L, Frédéric N, Soleiman Y.
Robust algebraic Schur complement preconditioners based on low rank corrections.
Inria research report. 2014;RR-8557:pp.18.

Remark: Consider having an implementation of the randomized set-up of Lorasc.

"""
function prepare_lorasc_precond(S::FunctionMap{Float64},
                                A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                                A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                                A_ΓΓ::SparseMatrixCSC{Float64,Int},
                                ind_Id_g2l::Array{Dict{Int,Int},1},
                                ind_Γ_g2l::Dict{Int,Int},
                                not_dirichlet_inds_g2l::Dict{Int,Int};
                                verbose=true,
                                compute_A_ΓΓ_chol=true,
                                nvec=25,
                                low_rank_correction=:exact,
                                ℓ=25,
                                ε=.01,
                                q=2)

  ndom, = size(A_IId)
  chol_A_IId = SuiteSparse.CHOLMOD.Factor{Float64}[]

  low_rank_correction == :randomized ? compute_A_ΓΓ_chol = true : nothing

  if compute_A_ΓΓ_chol
    ΠA_ΓΓ = nothing
    verbose ? print("compute cholesky factorization of A_ΓΓ ... ") : nothing
    time = @elapsed chol_A_ΓΓ = LinearAlgebra.cholesky(A_ΓΓ)
    verbose ? println("$time seconds") : nothing
  else
    verbose ? print("prepare amg for A_ΓΓ ... ") : nothing
    ΠA_ΓΓ = Preconditioners.AMGPreconditioner(A_ΓΓ)
    verbose ? println("$time seconds") : nothing
    chol_A_ΓΓ = nothing
  end

  verbose ? print("compute cholesky of A_IId's ... ") : nothing
  time = @elapsed for idom in 1:ndom
    push!(chol_A_IId, LinearAlgebra.cholesky(A_IId[idom]))
  end
  verbose ? println("$time seconds") : nothing

  # Compute low rank correction with exact generalized eigenpairs (σ, e) s.t. S * e == σ * A_ΓΓ * e
  if low_rank_correction == :exact
    verbose ? print("solve for general ld eigenpairs of (S, A_ΓΓ) ... ") : nothing
    time = @elapsed Σ, E, info = KrylovKit.geneigsolve(x_Γ -> (S * x_Γ, A_ΓΓ * x_Γ), 
                                                     A_ΓΓ.n, nvec, :SR, krylovdim=2*nvec, 
                                                     isposdef=true, ishermitian=true,
                                                     issymmetric=true)
    verbose ? println("$time seconds") : nothing

  # Compute low rank correction with approximate eigenpairs (ζ, u) s.t. L u 
  elseif low_rank_correction == :randomized
    verbose ? print("randomly approximate md eigenpairs of L^-1(A_ΓΓ - S)L^⁻T ... ") : nothing
    
    time = @elapsed begin
      H = Array{Float64,2}(undef, A_ΓΓ.n, ℓ)
      Q = Array{Float64,2}(undef, A_ΓΓ.n, ℓ)
      C = Array{Float64,2}(undef, ℓ, ℓ)

      Ξ = MvNormal(A_ΓΓ.n, 1.)
      H .= rand(Ξ, ℓ)

      for ivec in 1:ℓ
        for j in 1:2*q+1
          apply_bmat!(A_ΓΓ, chol_A_ΓΓ, S, H[:, ivec])
        end
      end

      Q .= Matrix(qr(H).Q)
      C .= Q'H

      Σ, E, info = KrylovKit.eigsolve(C)
      Σ .= 1 .- Σ
      for ivec in 1:ℓ
        E[ivec] .= chol_A_ΓΓ.L'(Q * E[ivec])
      end  

    end # time @elapsed
    verbose ? println("$time seconds") : nothing

  end # if low_rank_correction == :randomized

  # Order eigenpairs from least to more dominant
  println(Σ)
  order = sortperm(Σ)
  Σ .= Σ[order]
  E .= E[order]
  println(Σ)

  # Only keep necessary least dominant eigenpairs 
  nev = 0
  for (k, σ) in enumerate(Σ)
    if σ < ε
      Σ[k] = (ε - σ) / σ
      nev += 1
    else
      break
    end
  end 

  println(Σ)

  if nev == nvec
    println("Warning in prepare_lorasc_precond: nev == nvec -> pick a larger nvec.")
  elseif nev == 0 
    println("Warning in prepare_lorasc_precond: nev == 0 -> pick a larger ε.")
    nev = nvec
  end

  n_Γ = ind_Γ_g2l.count
  n = not_dirichlet_inds_g2l.count
  x_Id = [Array{Float64,1}(undef, ind_Id_g2l[idom].count) for idom in 1:ndom]
  x_Γ = Array{Float64, 1}(undef, n_Γ)
  z_Γ = Array{Float64, 1}(undef, n_Γ)
  u = Array{Float64,1}(undef, n)

  return LorascPreconditioner(A_IΓd,
                              chol_A_IId,
                              A_ΓΓ,
                              ΠA_ΓΓ,
                              chol_A_ΓΓ,
                              ε,
                              E[1:nev],
                              Σ[1:nev],
                              ind_Id_g2l,
                              ind_Γ_g2l,
                              not_dirichlet_inds_g2l,
                              x_Id,
                              x_Γ,
                              z_Γ,
                              u)
end


"""
     prepare_lorasc_precond(tentative_nnode::Int,
                            ndom::Int,
                            cells::Array{Int,2},
                            points::Array{Float64,2},
                            cell_neighbors::Array{Int,2},
                            a_vec::Array{Float64,1},
                            dirichlet_inds_g2l::Dict{Int,Int},
                            not_dirichlet_inds_g2l::Dict{Int,Int},
                            f::Function,
                            uexact::Function;
                            verbose=true,
                            do_local_schur_assembly=true,
                            load_partition=false,
                            compute_A_ΓΓ_chol=true,
                            nvec=25, 
                            ε=.01)

Prepares and returns a `LorascPreconditioner`.
See Grigori et al. (2014) for a reference on the LORASC preconditioner.

Grigori L, Frédéric N, Soleiman Y.
Robust algebraic Schur complement preconditioners based on low rank corrections.
Inria research report. 2014;RR-8557:pp.18.

Remark: There is no need to call both prepare_global_schur() and prepare_local_schurs().

"""
function prepare_lorasc_precond(tentative_nnode::Int,
                                ndom::Int,
                                cells::Array{Int,2},
                                points::Array{Float64,2},
                                cell_neighbors::Array{Int,2},
                                a_vec::Array{Float64,1},
                                dirichlet_inds_g2l::Dict{Int,Int},
                                not_dirichlet_inds_g2l::Dict{Int,Int},
                                f::Function,
                                uexact::Function;
                                verbose=true,
                                do_local_schur_assembly=false,
                                load_partition=false,
                                compute_A_ΓΓ_chol=true,
                                nvec=25, 
                                low_rank_correction=:randomized,
                                ℓ=25,
                                ε=.01)

  if load_partition
    epart, npart = load_partition(tentative_nnode, ndom)
  else
    epart, npart = mesh_partition(cells, ndom)
    save_partition(epart, npart, tentative_nnode, ndom)
  end

  ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l, node_owner,
  elemd, node_Γ, node_Γ_cnt, node_Id, nnode_Id = set_subdomains(cells,
                                                                cell_neighbors,
                                                                epart, 
                                                                npart,
                                                                dirichlet_inds_g2l)
  n_Γ = ind_Γ_g2l.count

  verbose ? print("prepare_global_schur ... ") : nothing
  time = @elapsed A_IId, A_IΓd, A_ΓΓ, b_Id, b_Γ = prepare_global_schur(cells,
                                                                       points,
                                                                       epart,
                                                                       ind_Id_g2l,
                                                                       ind_Γ_g2l,
                                                                       node_owner,
                                                                       a_vec,
                                                                       f,
                                                                       uexact)
  verbose ? println("$time seconds") : nothing


  verbose ? print("assemble amg preconditioners of A_IId ... ") : nothing
  time = @elapsed Π_IId = [AMGPreconditioner{SmoothedAggregation}(A_IId[idom])
                           for idom in 1:ndom];
  verbose ? println("$time seconds") : nothing

  verbose ? print("prepare_local_schurs ... ") : nothing
  time = @elapsed A_IIdd, A_IΓdd, A_ΓΓdd, _, _ = prepare_local_schurs(cells,
                                                                      points,
                                                                      epart,
                                                                      ind_Id_g2l,
                                                                      ind_Γd_g2l,
                                                                      ind_Γ_g2l,
                                                                      node_owner,
                                                                      a_vec,
                                                                      f,
                                                                      uexact)
  verbose ? println("$time seconds") : nothing

  if do_local_schur_assembly
    verbose ? print("assemble_local_schurs ... ") : nothing
    time = @elapsed Sd_local_mat = assemble_local_schurs(A_IIdd,
                                                         A_IΓdd,
                                                         A_ΓΓdd,
                                                         preconds=Π_IId)
    verbose ? println("$time seconds") : nothing


    S = LinearMap(x -> apply_local_schurs(Sd_local_mat,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x), nothing,
                                          n_Γ, issymmetric=true)

  else
    S = LinearMap(x -> apply_local_schurs(A_IIdd,
                                          A_IΓdd,
                                          A_ΓΓdd,
                                          ind_Γd_Γ2l,
                                          node_Γ_cnt,
                                          x,
                                          preconds=Π_IId), 
                                          nothing, n_Γ, issymmetric=true)
  end

  return prepare_lorasc_precond(S, A_IId, A_IΓd, A_ΓΓ, ind_Id_g2l,
                                ind_Γ_g2l, not_dirichlet_inds_g2l, 
                                compute_A_ΓΓ_chol=compute_A_ΓΓ_chol,
                                nvec=nvec, low_rank_correction=low_rank_correction,
                                ℓ=ℓ, ε=ε, verbose=verbose)

end


"""
     apply_lorasc(Πlorasc::LorascPreconditioner,
                  x::Array{Float64,1})

Applies the `LorascPreconditioner`.
  
"""
function apply_lorasc(Πlorasc::LorascPreconditioner,
                      x::Array{Float64,1})

  n, = size(x)
  ndom, = size(Πlorasc.ind_Id_g2l)
  n_Γ = Πlorasc.ind_Γ_g2l.count
  nvec, = size(Πlorasc.Σ)

  #x_Id = [Array{Float64,1}(undef, Πlorasc.ind_Id_g2l[idom].count) for idom in 1:ndom]
  #x_Γ = Array{Float64, 1}(undef, n_Γ)
  #z_Γ = Array{Float64, 1}(undef, n_Γ)
  #u = Array{Float64,1}(undef, n)

  for idom in 1:ndom
    for (node, node_in_I) in Πlorasc.ind_Id_g2l[idom]
      Πlorasc.x_Id[idom][node_in_I] = x[Πlorasc.not_dirichlet_inds_g2l[node]]
    end
  end

  for (node, node_in_Γ) in Πlorasc.ind_Γ_g2l
    val = x[Πlorasc.not_dirichlet_inds_g2l[node]]
    Πlorasc.x_Γ[node_in_Γ] = val
    Πlorasc.z_Γ[node_in_Γ] = val
  end

  for idom in 1:ndom
    Πlorasc.x_Id[idom] .= Πlorasc.chol_A_IId[idom] \ Πlorasc.x_Id[idom]
    Πlorasc.z_Γ .-= Πlorasc.A_IΓd[idom]' * Πlorasc.x_Id[idom]
  end

  if isnothing(Πlorasc.chol_A_ΓΓ)
    Πlorasc.x_Γ .= IterativeSolvers.cg(Πlorasc.A_ΓΓ, 
                                       Πlorasc.z_Γ,
                                       Pl=Πlorasc.ΠA_ΓΓ, tol=1e-9)
  else
    Πlorasc.x_Γ .= Πlorasc.chol_A_ΓΓ \ Πlorasc.z_Γ
  end

  for (k, σ) in enumerate(Πlorasc.Σ)
    val = Πlorasc.E[k]'Πlorasc.z_Γ
    Πlorasc.x_Γ .+= val * Πlorasc.E[k]
  end

  for idom in 1:ndom
    Πlorasc.x_Id[idom] .-= Πlorasc.chol_A_IId[idom] \ (Πlorasc.A_IΓd[idom] * Πlorasc.x_Γ)
  end

  for idom in 1:ndom
    for (node, node_in_I) in Πlorasc.ind_Id_g2l[idom]
      Πlorasc.u[Πlorasc.not_dirichlet_inds_g2l[node]] = Πlorasc.x_Id[idom][node_in_I]
    end
  end

  for (node, node_in_Γ) in Πlorasc.ind_Γ_g2l
    Πlorasc.u[Πlorasc.not_dirichlet_inds_g2l[node]] = Πlorasc.x_Γ[node_in_Γ]
  end

  return Πlorasc.u
end


import Base: \
function (\)(Πlorasc::LorascPreconditioner, x::Array{Float64,1})
  apply_lorasc(Πlorasc, x)
end

function LinearAlgebra.ldiv!(z::Array{Float64,1}, 
                             Πlorasc::LorascPreconditioner,
                             r::Array{Float64,1})
  z .= apply_lorasc(Πlorasc, r)
end

function LinearAlgebra.ldiv!(Πlorasc::LorascPreconditioner,
                             r::Array{Float64,1})
  r .= apply_lorasc(Πlorasc, copy(r))
end


struct DomainDecompositionLowRankPreconditioner

  # Li R & Saad Y. 
  # Low-rank correction methods for algebraic domain decomposition preconditioners. 
  # SIAM Journal on Matrix Analysis and Applications. 2017;38(3):807-28.

  A_IΓd::Array{SparseMatrixCSC{Float64,Int},1}
  chol_A0_Id::Array{SuiteSparse.CHOLMOD.Factor{Float64},1}
  A0_Γ::SparseMatrixCSC{Float64,Int}
  ΠA0_Γ
  α::Float64
  θ::Float64
  U::Array{Array{Float64,1},1}
  Λ::Array{Float64,1}
  ind_Id_g2l::Array{Dict{Int,Int},1}
  ind_Γ_g2l::Dict{Int,Int}
  not_dirichlet_inds_g2l::Dict{Int,Int}
end


"""
     apply_inv_a0!(chol_A0_Id::Array{SuiteSparse.CHOLMOD.Factor{Float64},1},
                   A0_Γ::SparseMatrixCSC{Float64,Int},
                   ΠA0_Γ,
                   x_Id::Array{Array{Float64,1},1},
                   x_Γ::Array{Float64,1})

Applies `A0` for the ddlr preconditioner. See Li & Saad (2017).

Li R & Saad Y. 
Low-rank correction methods for algebraic domain decomposition preconditioners. 
SIAM Journal on Matrix Analysis and Applications. 2017;38(3):807-28.

"""
function apply_inv_a0!(chol_A0_Id::Array{SuiteSparse.CHOLMOD.Factor{Float64},1},
                       A0_Γ::SparseMatrixCSC{Float64,Int},
                       ΠA0_Γ,
                       x_Id::Array{Array{Float64,1},1},
                       x_Γ::Array{Float64,1})

  ndom, = size(chol_A0_Id)
  for idom in 1:ndom
    x_Id[idom] .= chol_A0_Id[idom] \ x_Id[idom]
  end  
  x_Γ .= IterativeSolvers.cg(A0_Γ, x_Γ, Pl=ΠA0_Γ, tol=1e-12)
end


"""
     apply_inv_a0(chol_A0_Id::Array{SuiteSparse.CHOLMOD.Factor{Float64},1},
                  A0_Γ::SparseMatrixCSC{Float64,Int},
                  ΠA0_Γ,
                  x_Id::Array{Array{Float64,1},1},
                  x_Γ::Array{Float64,1})

Applies `A0` for the ddlr preconditioner. See Li & Saad (2017).

Li R & Saad Y. 
Low-rank correction methods for algebraic domain decomposition preconditioners. 
SIAM Journal on Matrix Analysis and Applications. 2017;38(3):807-28.
  
"""
function apply_inv_a0(chol_A0_Id::Array{SuiteSparse.CHOLMOD.Factor{Float64},1},
                      A0_Γ::SparseMatrixCSC{Float64,Int},
                      ΠA0_Γ,
                      x_Id::Array{Array{Float64,1},1},
                      x_Γ::Array{Float64,1})
  y_Id = copy(x_Id)
  y_Γ = copy(x_Γ)
  apply_inv_a0!(chol_A0_Id, A0_Γ, ΠA0_Γ, y_Id, y_Γ)
  return y_Id, y_Γ
end


"""
     apply_hmat(A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                chol_A0_Id::Array{SuiteSparse.CHOLMOD.Factor{Float64},1},
                A0_Γ::SparseMatrixCSC{Float64,Int},
                ΠA0_Γ,
                α::Float64,
                x_Γ::Array{Float64,1})

Applies `H` for the ddlr preconditioner. See Li & Saad (2017).

Li R & Saad Y. 
Low-rank correction methods for algebraic domain decomposition preconditioners. 
SIAM Journal on Matrix Analysis and Applications. 2017;38(3):807-28.
  
"""
function apply_hmat(A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                    chol_A0_Id::Array{SuiteSparse.CHOLMOD.Factor{Float64},1},
                    A0_Γ::SparseMatrixCSC{Float64,Int},
                    ΠA0_Γ,
                    α::Float64,
                    x_Γ::Array{Float64,1})

  ndom, = size(A_IΓd)

  z_Id = [Array{Float64,1}(undef, A_IΓd[idom].m) for idom in 1:ndom]
  z_Γ = copy(x_Γ)
  y_Γ = zeros(Float64, A0_Γ.n)

  # x = E * x_Γ
  for idom in 1:ndom
    z_Id[idom] .= α^-1 * (A_IΓd[idom] * x_Γ)
  end
  z_Γ .*= -α 

  # Solve A0 * y = x
  apply_inv_a0!(chol_A0_Id, A0_Γ, ΠA0_Γ, z_Id, z_Γ)

  # y_Γ = E' * y
  for idom in 1:ndom
    y_Γ .+= α^-1 * (A_IΓd[idom]' * z_Id[idom])
  end
  y_Γ .-= α * z_Γ

  return y_Γ
end


"""
     prepare_domain_decomposition_low_rank_precond(A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                                                   A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                                                   A_ΓΓ::SparseMatrixCSC{Float64,Int},
                                                   ind_Id_g2l::Array{Dict{Int,Int},1},
                                                   ind_Γ_g2l::Dict{Int,Int},
                                                   not_dirichlet_inds_g2l::Dict{Int,Int};
                                                   nvec=25,
                                                   α=1.)

Prepares and returns a DomainDecompositionLowRankPreconditioner.
See Li & Saad (2017) for a reference on the domain decomposition
low rank (ddlr) preconditioner. 

Notice: This preconditioner is meant for a partition with vertex separators.
        Work to be resumed if an edge-based partitioning domain decomposition
        is implemented.

Li R & Saad Y. 
Low-rank correction methods for algebraic domain decomposition preconditioners. 
SIAM Journal on Matrix Analysis and Applications. 2017;38(3):807-28.
  
"""
function prepare_domain_decomposition_low_rank_precond(A_IId::Array{SparseMatrixCSC{Float64,Int},1},
                                                       A_IΓd::Array{SparseMatrixCSC{Float64,Int},1},
                                                       A_ΓΓ::SparseMatrixCSC{Float64,Int},
                                                       ind_Id_g2l::Array{Dict{Int,Int},1},
                                                       ind_Γ_g2l::Dict{Int,Int},
                                                       not_dirichlet_inds_g2l::Dict{Int,Int};
                                                       nvec=25,
                                                       α=1.)
  
  ndom, = size(A_IId)
  chol_A0_Id = SuiteSparse.CHOLMOD.Factor{Float64}[]

  A0_Γ = A_ΓΓ + α^2 * LinearAlgebra.I
  ΠA0_Γ = Preconditioners.AMGPreconditioner(A0_Γ)

  for idom in 1:ndom
    A0_I = A_IId[idom] + α^-2 * A_IΓd[idom] * A_IΓd[idom]'
    push!(chol_A0_Id, LinearAlgebra.cholesky(A0_I))
  end

  Λ, U, info = KrylovKit.eigsolve(x_Γ -> apply_hmat(A_IΓd, chol_A0_Id, A0_Γ, ΠA0_Γ, α, x_Γ),
                               A_ΓΓ.n, nvec+1, :LM, issymmetric=true, krylovdim=2*nvec)
  θ = Λ[nvec + 1]
  
  return DomainDecompositionLowRankPreconditioner(A_IΓd,
                                                  chol_A0_Id,
                                                  A0_Γ,
                                                  ΠA0_Γ,
                                                  α,
                                                  θ,
                                                  U[1:nvec],
                                                  Λ[1:nvec],
                                                  ind_Id_g2l,
                                                  ind_Γ_g2l,
                                                  not_dirichlet_inds_g2l)
end


"""
     apply_domain_decomposition_low_rank(Πddlr::DomainDecompositionLowRankPreconditioner,
                                         x::Array{Float64,1})

Applies the `DomainDecompositionLowRankPreconditioner`.
  
"""
function apply_domain_decomposition_low_rank(Πddlr::DomainDecompositionLowRankPreconditioner,
                                             x::Array{Float64,1})

  n, = size(x)
  ndom, = size(Πddlr.ind_Id_g2l)
  n_Γ = Πddlr.ind_Γ_g2l.count
  nvec, = size(Πddlr.Λ)

  x_Id = [Array{Float64,1}(undef, Πddlr.ind_Id_g2l[idom].count) for idom in 1:ndom]
  x_Γ = Array{Float64, 1}(undef, n_Γ)
  y_Γ = zeros(Float64, n_Γ)
  w_Γ = Array{Float64,1}(undef, n_Γ)
  u = Array{Float64,1}(undef, n)

  for idom in 1:ndom
    for (node, node_in_I) in Πddlr.ind_Id_g2l[idom]
      x_Id[idom][node_in_I] = x[Πddlr.not_dirichlet_inds_g2l[node]]
    end
  end

  for (node, node_in_Γ) in Πddlr.ind_Γ_g2l
    x_Γ[node_in_Γ] = x[Πddlr.not_dirichlet_inds_g2l[node]]
  end
    
  # Solve A0 * z = x
  z_Id, z_Γ = apply_inv_a0(Πddlr.chol_A0_Id, 
                           Πddlr.A0_Γ,
                           Πddlr.ΠA0_Γ,
                           x_Id,
                           x_Γ)

  # y_Γ = E' * z
  for idom in 1:ndom
    y_Γ .+= Πddlr.α^-1 * (Πddlr.A_IΓd[idom]' * z_Id[idom])
  end
  y_Γ .-= Πddlr.α * z_Γ

  # w_Γ = Ginv_approx * y_Γ
  w_Γ .= y_Γ ./ (1 - Πddlr.θ)
  for k in 1:nvec
    val = Πddlr.U[k]'y_Γ
    val *= (1 - Πddlr.Λ[k])^-1 - (1 - Πddlr.θ)^-1
    w_Γ .+= val * Πddlr.U[k]
  end

  # x = x + v with v = E * w_Γ
  for idom in 1:ndom
    x_Id[idom] .+= Πddlr.α^-1 * (Πddlr.A_IΓd[idom] * w_Γ)
  end
  x_Γ .-= Πddlr.α * w_Γ

  # Solve A0 u = x
  apply_inv_a0!(Πddlr.chol_A0_Id,
                Πddlr.A0_Γ, 
                Πddlr.ΠA0_Γ,
                x_Id,
                x_Γ)

  for idom in 1:ndom
    for (node, node_in_I) in Πddlr.ind_Id_g2l[idom]
      u[Πddlr.not_dirichlet_inds_g2l[node]] = x_Id[idom][node_in_I]
    end
  end

  for (node, node_in_Γ) in Πddlr.ind_Γ_g2l
    u[Πddlr.not_dirichlet_inds_g2l[node]] = x_Γ[node_in_Γ]
  end

  return u
end


import Base: \
function (\)(Πddlr::DomainDecompositionLowRankPreconditioner, x::Array{Float64,1})
  apply_domain_decomposition_low_rank(Πddlr, x)
end

function LinearAlgebra.ldiv!(z::Array{Float64,1}, 
                             Πddlr::DomainDecompositionLowRankPreconditioner,
                             r::Array{Float64,1})
  z .= apply_domain_decomposition_low_rank(Πddlr, r)
end

function LinearAlgebra.ldiv!(Πddlr::DomainDecompositionLowRankPreconditioner,
                             r::Array{Float64,1})
  r .= apply_domain_decomposition_low_rank(Πddlr, copy(r))
end


struct NeumannNeumannInducedPreconditioner
  ΠSd::Array{Array{Float64,2},1}
  A_IIdd::Array{SparseMatrixCSC{Float64,Int},1}
  chol_A_IId::Array{SuiteSparse.CHOLMOD.Factor{Float64},1}
  A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1}
  ind_Id_g2l::Array{Dict{Int,Int},1}
  ind_Γ_g2l::Dict{Int,Int}
  ind_Γd_Γ2l::Array{Dict{Int,Int},1}
  node_Γ_cnt::Array{Int,1}
  node_Γ::Array{Int,1}
  not_dirichlet_inds_g2l::Dict{Int,Int}
end


"""
     prepare_neumann_neumann_induced_precond(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                                             A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                             A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                             ind_Id_g2l::Array{Dict{Int,Int},1},
                                             ind_Γ_g2l::Dict{Int,Int},
                                             ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                                             node_Γ_cnt::Array{Int,1},
                                             node_Γ::Array{Int,1},
                                             not_dirichlet_inds_g2l::Dict{Int,Int};
                                             preconds=nothing)

Prepares and returns a `NeumannNeumannInducedPreconditioner`.

Notice: This preconditioner only seems to work with deflation.
  
"""
function prepare_neumann_neumann_induced_precond(A_IIdd::Array{SparseMatrixCSC{Float64,Int},1},
                                                 A_IΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                                 A_ΓΓdd::Array{SparseMatrixCSC{Float64,Int},1},
                                                 ind_Id_g2l::Array{Dict{Int,Int},1},
                                                 ind_Γ_g2l::Dict{Int,Int},
                                                 ind_Γd_Γ2l::Array{Dict{Int,Int},1},
                                                 node_Γ_cnt::Array{Int,1},
                                                 node_Γ::Array{Int,1},
                                                 not_dirichlet_inds_g2l::Dict{Int,Int};
                                                 preconds=nothing)

  ndom, = size(A_IIdd)
  ΠSd = Array{Float64,2}[]
  chol_A_IId = SuiteSparse.CHOLMOD.Factor{}[]

  for idom in 1:ndom
    if isnothing(preconds)
      Sd = LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                             A_IΓdd[idom],
                                             A_ΓΓdd[idom],
                                             xd,
                                             reltol=1e-12),
                                             ind_Γd_Γ2l[idom].count, issymmetric=true)
    else
      Sd = LinearMap(xd -> apply_local_schur(A_IIdd[idom],
                                             A_IΓdd[idom],
                                             A_ΓΓdd[idom],
                                             xd,
                                             precond=preconds[idom],
                                             reltol=1e-12),
                                             ind_Γd_Γ2l[idom].count, issymmetric=true)
    end

    Sd_mat = Array(Sd)
    push!(ΠSd, LinearAlgebra.pinv(Sd_mat, rtol=sqrt(eps(real(float(one(eltype(Sd_mat))))))))
    push!(chol_A_IId, LinearAlgebra.cholesky(A_IIdd[idom]))
  end

  return NeumannNeumannInducedPreconditioner(ΠSd,
                                             A_IIdd,
                                             chol_A_IId,
                                             A_IΓdd,
                                             ind_Id_g2l,
                                             ind_Γ_g2l,
                                             ind_Γd_Γ2l,
                                             node_Γ_cnt,
                                             node_Γ,
                                             not_dirichlet_inds_g2l)
end


"""
     apply_neumann_neumann_induced(Πnn::NeumannNeumannInducedPreconditioner,
                                   r::Array{Float64,1})
  
Applies the `NeumannNeumannInducedPreconditioner`.
  
"""
function apply_neumann_neumann_induced(Πnn::NeumannNeumannInducedPreconditioner,
                                       r::Array{Float64,1})
  
  n, = size(r)  
  ndom, = size(Πnn.ΠSd)
  n_Γ, = size(Πnn.node_Γ_cnt)

  n_Id = [Πnn.A_IIdd[idom].n for idom in 1:ndom]
  r_Id = [Array{Float64,1}(undef, n_Id[idom]) for idom in 1:ndom]
  z_Id = [Array{Float64,1}(undef, n_Id[idom]) for idom in 1:ndom]

  n_Γd = [Πnn.ind_Γd_Γ2l[idom].count for idom in 1:ndom]
  r_Γd = [Array{Float64,1}(undef, n_Γd[idom]) for idom in 1:ndom]
  z_Γd = [Array{Float64,1}(undef, n_Γd[idom]) for idom in 1:ndom]

  r_schur = Array{Float64,1}(undef, n_Γ)
  r_Γ = Array{Float64,1}(undef, n_Γ)
  z_Γ = zeros(Float64, n_Γ)
  z = Array{Float64,1}(undef, n)

  for idom in 1:ndom
    for (node, node_in_Id) in Πnn.ind_Id_g2l[idom]
      r_Id[idom][node_in_Id] = r[Πnn.not_dirichlet_inds_g2l[node]]
    end
  end

  for (node, node_in_Γ) in Πnn.ind_Γ_g2l
    r_Γ[node_in_Γ] = r[Πnn.not_dirichlet_inds_g2l[node]]
  end

  r_schur .= r_Γ
  for idom in 1:ndom
    z_Id[idom] .= Πnn.chol_A_IId[idom] \ r_Id[idom]
    r_Γd[idom] .= Πnn.A_IΓdd[idom]' * z_Id[idom]
    for (node_in_Γ, node_in_Γd) in Πnn.ind_Γd_Γ2l[idom]
      r_schur[node_in_Γ] -= r_Γd[idom][node_in_Γd]
    end
  end

  for idom in 1:ndom
    for (node_in_Γ, node_in_Γd) in Πnn.ind_Γd_Γ2l[idom]
      r_Γd[idom][node_in_Γd] = r_schur[node_in_Γ] / Πnn.node_Γ_cnt[node_in_Γ]
    end
    z_Γd[idom] .= Πnn.ΠSd[idom] * r_Γd[idom]

    for (node_in_Γ, node_in_Γd) in Πnn.ind_Γd_Γ2l[idom]
      z_Γ[node_in_Γ] += z_Γd[idom][node_in_Γd] / Πnn.node_Γ_cnt[node_in_Γ]
    end
    z_Id[idom] .= r_Id[idom] .- Πnn.A_IΓdd[idom] * z_Γd[idom]
    z_Id[idom] .= Πnn.chol_A_IId[idom] \ z_Id[idom]
    for (node, node_in_Id) in Πnn.ind_Id_g2l[idom]
      z[Πnn.not_dirichlet_inds_g2l[node]] = z_Id[idom][node_in_Id]
    end
  end # for idom

  for (node, node_in_Γ) in Πnn.ind_Γ_g2l
    z[Πnn.not_dirichlet_inds_g2l[node]] = z_Γ[node_in_Γ] 
  end

  return z
end


import Base: \
function (\)(Πnn::NeumannNeumannInducedPreconditioner, x::Array{Float64,1})
  apply_neumann_neumann_induced(Πnn, x)
end

function LinearAlgebra.ldiv!(z::Array{Float64,1}, 
                             Πnn::NeumannNeumannInducedPreconditioner,
                             r::Array{Float64,1})
  z .= apply_neumann_neumann_induced(Πnn, r)
end

function LinearAlgebra.ldiv!(Πnn::NeumannNeumannInducedPreconditioner,
                             r::Array{Float64,1})
  r .= apply_neumann_neumann_induced(Πnn, copy(r))
end