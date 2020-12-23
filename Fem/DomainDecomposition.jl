import TriangleMesh
using DelimitedFiles

function mesh_partition(mesh::TriangleMesh.TriMesh, nd::Int)
    nel = mesh.n_cell # number of elements
  
    # Write mesh for mpmetis
    open("mesh.metis", "w") do io
      print(io, "$nel\n")
      for el in 1:nel
        print(io, "$(mesh.cell[1, el]) $(mesh.cell[2, el]) $(mesh.cell[3, el])\n")
      end
    end
  
    # Call mpmetis for contiguous partition
    run(`mpmetis mesh.metis $nd -contig`, wait=true)
    epart = readdlm("mesh.metis.epart.$nd", Int)
    npart = readdlm("mesh.metis.npart.$nd", Int)
  
    return epart, npart 
  end
  
  function set_subdomains(mesh::TriangleMesh.TriMesh, epart::Array{Int64,2})
    nel = mesh.n_cell # Number of elements
  
    # Correct potential indexing error of TriangleMesh.jl
    bnd_tag, iel_max = extrema(mesh.cell_neighbor)
    if iel_max > nel
      mesh.cell_neighbor .-= 1
      bnd_tag -= 1
    end
  
    iel_cell, jel_cell = zeros(Int, 3), zeros(Int, 3)
    nodes_at_interface = Int[]
  
    # Loop over elements in mesh
    for iel in 1:nel
  
      # Get global nodes and subdomain of element
      iel_cell = mesh.cell[:, iel]
      iel_dom = epart[iel]
  
      # Loop over neighbors
      for j in 1:3
  
        # Check if it is not a boundary segment 
        jel = mesh.cell_neighbor[j, iel]
        if jel != bnd_tag
          
          # Check if neighbor belongs to another subdomain
          jel_dom = epart[jel]
          if jel_dom != iel_dom
            jel_cell = mesh.cell[:, jel]
  
            # Store common nodes in nodes_at_interface
            for node in iel_cell
              if (node in jel_cell) & !(node in nodes_at_interface)
                push!(nodes_at_interface, node)
              end
            end
          end
        end
      end
    end
    return nodes_at_interface
  end
  