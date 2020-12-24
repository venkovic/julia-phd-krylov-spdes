import TriangleMesh
using DelimitedFiles

function mesh_partition(mesh::TriangleMesh.TriMesh, nsbd::Int)
    nel = mesh.n_cell # number of elements
  
    # Write mesh for mpmetis
    open("mesh.metis", "w") do io
      print(io, "$nel\n")
      for el in 1:nel
        # Metis starts indexing nodes at 1 
        print(io, "$(mesh.cell[1, el]) $(mesh.cell[2, el]) $(mesh.cell[3, el])\n")
      end
    end
  
    # Call mpmetis for contiguous partition
    run(`mpmetis mesh.metis $nsbd -contig`, wait=true)
    epart = readdlm("mesh.metis.epart.$nsbd", Int) .+ 1 # Metis starts indexing doms at 0
    npart = readdlm("mesh.metis.npart.$nsbd", Int) .+ 1 # Metis starts indexing doms at 0
  
    return epart, npart 
  end
  
function set_subdomains(mesh::TriangleMesh.TriMesh, epart::Array{Int64,2}, npart::Array{Int64,2})
    nel = mesh.n_cell # Number of elements
    nn = mesh.n_point # Number of nodes

    ndom = maximum(epart) # Number of subdomains
    elemd = [Int[] for _ in 1:ndom] # Elements for each subdomain 
    node_Γ = Int[] # Nodes at the interface of the partition
    node_Id = [Int[] for _ in 1:ndom] # Inside nodes of each subdomain
    ind_Id = [Dict{Int, Int}() for _ in 1:ndom] # Conversion table to interior indexes of each subdomain
    nn_Id = zeros(Int, ndom) # Number of interior nodes for each subdomain
    is_on_Γ = zeros(Bool, nn) # Says if a node is on the partition's interface 

    # Correct potential indexing error of TriangleMesh.jl
    bnd_tag, iel_max = extrema(mesh.cell_neighbor)
    if iel_max > nel
      mesh.cell_neighbor .-= 1
      bnd_tag -= 1
    end
  
    # Loop over elements in mesh
    iel_cell, jel_cell = zeros(Int, 3), zeros(Int, 3)
    for iel in 1:nel
  
      # Get global nodes and subdomain of element
      iel_cell = mesh.cell[:, iel]
      id = epart[iel]
      push!(elemd[id], iel)
  
      # Loop over neighbors
      for j in 1:3
  
        # If this is not a boundary segment 
        jel = mesh.cell_neighbor[j, iel]
        if jel != bnd_tag
          
          # If neighbor belongs to another subdomain
          jd = epart[jel]
          if jd != id
            jel_cell = mesh.cell[:, jel]
  
            # Store common nodes in nodes_Γ
            for node in iel_cell
              if (node in jel_cell) & !(node in node_Γ)
                push!(node_Γ, node)
                is_on_Γ[node] = true
              end
            end
          end
        end
      end
    end

    # Store inside nodes for each subdomain
    for inode in 1:nn
      if !(is_on_Γ[inode])
        id = npart[inode]
        inode in node_Id[id] ? nothing : push!(node_Id[id], inode) 
      end
    end

    # Indexing of interior for each subdomain 
    for id in 1:ndom
      nn_Id[id] = length(node_Id[id])
      for i in 1:nn_Id[id]
        ind_Id[id][node_Id[id][i]] = i
      end
    end

    return elemd, node_Γ, node_Id, ind_Id, nn_Id, is_on_Γ
end

function do_IId_assembly(elemd, cell, point, is_on_Γ, ind_Id, nn_Id, a, f, idom)
  nnd = nn_Id[idom]
  A_IId = zeros(Float64, nnd, nnd)
  for (i, iel) in enumerate(elemd[idom])
    iel_cell = mesh.cell[:, iel]
    # Usual loop over nodes, with following conditions for BOTH nodes
    r, s = 1, 1
    if !(is_on_Γ[r])
      if !(is_on_Γ[s])
        A_IId[ind_Id[r], Ind_Id[s]] += 1.
      end
    end
  end
  return A_IId
end

function do_IΓ_assembly()
  A_IΓ = nothing
  return A_IΓ
end

function do_ΓΓ_assembly()
  A_IΓ = nothing
  return A_IΓ
end