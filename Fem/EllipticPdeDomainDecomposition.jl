import TriangleMesh
using DelimitedFiles
using IterativeSolvers
  

"""
set_subdomains(mesh::TriangleMesh.TriMesh, epart::Array{Int64,2}, npart::Array{Int64,2})
  
Returns helper data structures for non-overlaping domain decomposition using the mesh 
partition defined by epart and npart. 
  
Input:
  
mesh: Instance of TriangleMesh.TriMesh.

nel = mesh.n_cell
epart[iel, 1]: subdomain idom ∈ [1, ndom] to which element iel ∈ [1, nel] belongs.

nn = mesh.n_point
npart[inode, 1]: subdomain idom ∈ [1, ndom] to which node inode ∈ [1, nn] belongs.

Output:

elemd[idom][:]: 1D array of all the elements contained in subdomain idom ∈ [1, ndom].

node_Γ[:]: 1D array of global indices of the nodes at the interface of the mesh
           partition.


           
           ? IS node_Id necessary ?


node_Id[idom]: 1D array of global indices of the nodes strictly inside subdomain 
               idom ∈ [1, ndom].

ind_Id[idom]: Conversion table (i.e. dictionary) from global to local indices of 
              nodes strictly inside subdomain idom ∈ [1, ndom].

nn_Id[idom]: Number of nodes strictly inside subdomain idom ∈ [1, ndom].

is_on_Γ[inode]: True if node inode ∈ [1, nn] is on the interface of the mesh 
                partition, and false otherwise.
  
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
elemd, node_Γ, node_Id, ind_Id, nn_Id, is_on_Γ = set_subdomains(mesh, epart, npart)

```
"""
function set_subdomains(mesh::TriangleMesh.TriMesh, epart::Array{Int64,2}, npart::Array{Int64,2})
    nel = mesh.n_cell # Number of elements
    nn = mesh.n_point # Number of nodes
    ndom = maximum(epart) # Number of subdomains

    # Essential data structures
    ind_Id = [Dict{Int, Int}() for _ in 1:ndom] # Conversion table from global to local indices of 
                                                # nodes strictly inside each subdomain
    ind_Γ = Dict{Int, Int}() # Conversion table from global to local indices of 
                             # nodes on the interface of the mesh partition
    is_on_Γ = zeros(Bool, nn) # Says if a node is on the interface of the mesh partition, or not 

    # Extra data structures
    elemd = [Int[] for _ in 1:ndom] # Elements contained inside each subdomain 
    node_Γ = Int[] # Nodes at the interface of the mesh partition
    node_Id = [Int[] for _ in 1:ndom] # Nodes strictly inside each subdomain
    nn_Id = zeros(Int, ndom) # Number of nodes strictly inside each subdomain

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
      idom = epart[iel]
      push!(elemd[idom], iel)
  
      # Loop over neighbors
      for j in 1:3
  
        # If this is not a boundary segment 
        jel = mesh.cell_neighbor[j, iel]
        if jel != bnd_tag
          
          # If neighbor belongs to another subdomain
          jdom = epart[jel]
          if jdom != idom
            jel_cell = mesh.cell[:, jel]
  
            # Store common nodes in nodes_Γ
            for node in iel_cell
              if (node in jel_cell) & !(node in node_Γ)
                push!(node_Γ, node)
                is_on_Γ[node] = true
                ind_Γ[node] = ind_Γ.count + 1
              end
            end
          end
        end
      end
    end

    # Store inside nodes for each subdomain
    for inode in 1:nn
      if !(is_on_Γ[inode])
        idom = npart[inode]
        inode in node_Id[idom] ? nothing : push!(node_Id[idom], inode) 
      end
    end

    # Indexing of interior for each subdomain 
    for idom in 1:ndom
      nn_Id[idom] = length(node_Id[idom])
      for i in 1:nn_Id[idom]
        ind_Id[idom][node_Id[idom][i]] = i
      end
    end

    return ind_Id, ind_Γ, is_on_Γ, elemd, node_Γ, node_Id, nn_Id
end


#function do_schur_assembly(elemsd, cells, points, is_on_Γ, ind_Id, nn_Id, a, f)


function do_schur_assembly(cells, points, epart, ind_Id, ind_Γ, is_on_Γ, a, f)
  ndom = length(ind_Id) # Number of subdomains  
  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of nodes
  n_Γ = ind_Γ.count # Number of nodes on the interface of the mesh partition
  #b_vec = zeros(nnode, 1) # Right hand side
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
  
  # Indices (I, J) and data (V) for sparse Galerkin operator A_ΓΓ
  ΓΓ_I, ΓΓ_J, ΓΓ_V =  Int[], Int[], Float64[] 

  # Right hand sides
  b_I, b_Γ = Int[], Int[]

  # Loop over elements
  for iel in 1:nel
    # Get subdomain of element
    idom = epart[iel]

    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    coeff = 0.
    for j in 1:3
      jj = cells[j, iel]
      x[j], y[j] = points[1, jj], points[2, jj]
      coeff += a(x[j], y[j])
    end
    coeff /= 3.
    
    # Terms of the shoelace formula for a triangle
    Δx[1] = x[3] - x[2]
    Δx[2] = x[1] - x[3]
    Δx[3] = x[2] - x[1]
    Δy[1] = y[2] - y[3]
    Δy[2] = y[3] - y[1]
    Δy[3] = y[1] - y[2]
    
    # Area of element
    Area = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
    
    # Loop over vertices of element
    for i in 1:3
      ii = cells[i, iel]
      ii_is_on_Γ = is_on_Γ[ii]

      # Loop over vertices of element
      for j in 1:3

        # Store local contribution
        Kij = coeff * (Δy[i] * Δy[j] + Δx[i] * Δx[j]) / 4 / Area
        jj = cells[j, iel]

        if ii_is_on_Γ
          if is_on_Γ[jj]

            # Both nodes are on the interface of the mesh partition, 
            # Add contribution to A_ΓΓ:
            push!(ΓΓ_I, ind_Γ[ii])
            push!(ΓΓ_J, ind_Γ[jj])
            push!(ΓΓ_V, Kij)
          end
        else
          if is_on_Γ[jj]

            # First node is inside subdomain idom, and second is on the interface 
            # of the mesh partition,
            # Add contribution to A_IΓ:
            push!(IΓd_I[idom], ind_Id[idom][ii])
            push!(IΓd_J[idom], ind_Γ[jj])
            push!(IΓd_V[idom], Kij)
          else

            # Both nodes are strictly inside the subdomain idom,
            # Add contribution to A_IΓ:
            push!(IId_I[idom], ind_Id[idom][ii])
            push!(IId_J[idom], ind_Id[idom][jj])
            push!(IId_V[idom], Kij)
          end
        end
      end
    end
  end

  # Assemble csc sparse arrays
  A_IId = [sparse(IId_I[idom], IId_J[idom], IId_V[idom]) for idom in 1:ndom]
  A_IΓd = [sparse(IΓd_I[idom], IΓd_J[idom], IΓd_V[idom], size(A_IId[idom])[1], n_Γ) for idom in 1:ndom]
  A_ΓΓ = sparse(ΓΓ_I, ΓΓ_J, ΓΓ_V)

  return A_IId, A_IΓd, A_ΓΓ, b_I, b_Γ
end


function apply_schur(A_IId, A_IΓd, A_ΓΓ, x)
  ndom = length(A_IId)
  Sx = A_ΓΓ * x
  for idom in 1:ndom
    v = IterativeSolvers.cg(A_IId[idom], A_IΓd[idom] * x)
    Sx .-= A_IΓd[idom]' * v
  end
  return Sx
end