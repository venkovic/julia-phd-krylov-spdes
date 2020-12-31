using SparseArrays
import TriangleMesh


"""
set_subdomain(mesh::TriangleMesh.TriMesh, epart::Array{Int64,2}, npart::Array{Int64,2})
  
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
function set_subdomain(mesh::TriangleMesh.TriMesh, epart::Array{Int64,2}, idom::Int)
  nel = mesh.n_cell # Number of elements
  inds_l2g = Int[]  # Conversion table from local to global indices of the
                    # nodes of the idom-th subdomain
  inds_g2l = Dict{Int, Int}() # Conversion table from global to local indices of the 
                              # nodes of the idom-th subdomain
  elems = Int[] # Elements contained inside each subdomain

  iel_cell = zeros(Int, 3)
  for iel in 1:nel

    # Element is in subdomain idom
    if epart[iel] == idom
      
      # Add element
      push!(elems, iel)

      # Set-up local indices and conversion tables for nodes in element
      for node in mesh.cell[:, iel]
        if !haskey(inds_g2l, node)
          push!(inds_l2g, node)
          inds_g2l[node] = length(inds_l2g)
        end
      end
    end
  end

  return inds_l2g, inds_g2l, elems
end


"""
do_local_mass_covariance_assembly(cells, points, inds_l2g, inds_g2l, cov)

Assembles local Galerkin kernel operator of a subdomain generated by the
covariance function cov with 2D P1 finite elements with a given global
triangulation (cells, points).

Input:

cells[1:3, :]: Global node indices of each node of each element of the mesh
cells[1, el]: Global node index of 1st node of the el-th element
cells[2, el]: Global node index of 2nd node of the el-th element
cells[3, el]: Global node index of 3rd node of the el-th element

points[1:2, :]: Coordinates of all nodal points of the mesh
points[1, node]: x-coordinate of global node
points[2, node]: y-coordinate of global node

inds_l2g[inode]: Global node index of local inode

inds_g2l[node]: Local node index of global node

cov: function(x1::Float64, y1::Float64, x2::Float64, y2::Float64)::Float64 

Output:

C: Array of Galerkin formulation (nnode-by-nnode) where nnode = inds_g2l.count
   with components C_ij = ∫_Ω ϕ_i(P) ∫_{Ω'} cov(P, P') ϕ_j(P') dΩ' dΩ.

# Examples
```jldoctest
julia>
using TriangleMesh
using Fem

poly = polygon_Lshape()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  L = .1
  return exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

# Assembly for 11_894 DoFs
C = @time do_mass_covariance_assembly(mesh.cell, mesh.point, cov)

Maximum triangle area: .00005
 90.211575 seconds (243.67 k allocations: 2.121 GiB, 0.01% gc time)

```
"""
function do_local_mass_covariance_assembly(cells, points, inds_l2g, inds_g2l, elems, cov)
  nel = length(elems) # Number of elements in subdomain
  nnode = inds_g2l.count # Number of nodes of subdomain
  R = zeros(nnode, nnode) # R[i, j] ≈ ∑_e ∫_{Ω'_e} ϕ_i(P') cov(P', P_j) dΩ'
  C = zeros(nnode, nnode) # C[i, j] ≈ ∫_Ω ϕ_i(P) ∫_{Ω'} cov(P, P') ϕ_j(P') dΩ' dΩ.
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3) # Used to store terms of shoelace formula
  Area = zeros(nel) # Used to store element areas
  
  # Loop over mesh nodes
  for (jnode, node) in enumerate(inds_l2g)
    
    # Get coordinates of node
    xj = points[1, node]
    yj = points[2, node]
    
    # Loop over elements
    for (iel, el) in enumerate(elems)
      
      # Get (x, y) coordinates of each element vertex
      # and coefficient at the center of the element
      for r in 1:3
        rr = cells[r, el]
        x[r], y[r] = points[1, rr], points[2, rr]
      end
      
      # Terms of the shoelace formula for a triangle
      Δx[1] = x[3] - x[2]
      Δx[2] = x[1] - x[3]
      Δx[3] = x[2] - x[1]
      Δy[1] = y[2] - y[3]
      Δy[2] = y[3] - y[1]
      Δy[3] = y[1] - y[2]
      
      # Area of element
      Area_el = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
      Area[iel] = Area_el
      
      # Add local contributions
      for r in 1:3
        s = r + 1 - floor(Int, (r + 1) / 3) * 3
        s == 0 ? s = 3 : nothing
        t = r + 2 - floor(Int, (r + 2) / 3) * 3
        t == 0 ? t = 3 : nothing
        inode = inds_g2l[cells[r, el]]
        R[inode, jnode] += (2 * cov(x[r], y[r], xj, yj) 
                              + cov(x[s], y[s], xj, yj) 
                              + cov(x[t], y[t], xj, yj)) * Area_el / 12
      end
    end
  end
  
  # Loop over mesh nodes
  for (inode, node) in enumerate(inds_l2g)
    
    # Loop over elements
    for (jel, el) in enumerate(elems)
      
      # Get area of element
      Area_el = Area[jel]
      
      # Add local contributions
      for r in 1:3
        s = r + 1 - floor(Int, (r + 1) / 3) * 3
        s == 0 ? s = 3 : nothing
        t = r + 2 - floor(Int, (r + 2) / 3) * 3
        t == 0 ? t = 3 : nothing
        jnode = inds_g2l[cells[r, el]]
        knode = inds_g2l[cells[s, el]]
        lnode = inds_g2l[cells[t, el]]
        C[inode, jnode] += (2 * R[inode, jnode] 
                              + R[inode, knode]  
                              + R[inode, lnode]) * Area_el / 12
      end
    end
  end
  
  return C
end


"""
do_local_mass_assembly(cells, points)

Gets mass matrix of linear form using 2D P1 finite elements with a 
given triangulation (cells, points).

Input:

cells[1:3, 1:nel]: Global node indices of local nodes in each element
cells[1, iel]: Global node index of 1st node of element iel ∈ [1, nel]
cells[2, iel]: Global node index of 2nd node of element iel ∈ [1, nel]
cells[3, iel]: Global node index of 3rd node of element iel ∈ [1, nel]

points[1:2, 1:nnode]: Coordinates of all nodal points
points[1, inode]: x-coordinate of global node indexed inode ∈ [1, nnode]
points[2, inode]: y-coordinate of global node indexed inode ∈ [1, nnode]

Output:

M: Masss matrix formulation (nnode-by-nnode) with components 
   M_ij = ∫_Ω ϕ_i(P) ϕ_j(P) dΩ.

# Examples
```jldoctest
julia>
using TriangleMesh
using Fem

poly = polygon_Lshape()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

# Assembly for 11_894 DoFs
M = @time get_mass_matrix(mesh.cell, mesh.point)

Maximum triangle area: .00005
  0.337066 seconds (595.71 k allocations: 44.153 MiB)

```
"""
function do_local_mass_assembly(cells, points, inds_g2l, elems)
  nel = length(elems) # Number of elements in subdomain
  nnode = inds_g2l.count # Number of nodes of subdomain
  I, J, V = Int[], Int[], Float64[] # Indices (I, J) and data (V) for mass matrix
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3) # Used to store terms of shoelace formula
  
  # Loop over elements
  for (iel, el) in enumerate(elems)
    
    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    for r in 1:3
      rnode = cells[r, el]
      x[r], y[r] = points[1, rnode], points[2, rnode]
    end
    
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
      inode = inds_g2l[cells[i, el]]

      # Loop over vertices of element
      for j in 1:3
        
        # Store local contribution
        if i == j
          Kij = Area / 6.
        else
          Kij = Area / 12.
        end
        jnode = inds_g2l[cells[j, el]]
        push!(I, inode)
        push!(J, jnode)
        push!(V, Kij)
      end
    end
  end
  
  # Assemble sparse mass matrix
  M = sparse(I, J, V)
  
  return M
end



function do_global_mass_reduced_assembly(cells, points, epart::Array{Int64,2}, inds_g2ld, Φd)
    _, nel = size(cells) # Number of elements
    ndom = length(Φd) # Number of subdomains
    md = Int[] # Number of local modes for each subdomain
    I, J, V = Int[], Int[], Float64[] # Indices (I, J) and data (V) for mass matrix
    x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
    Δx, Δy = zeros(3), zeros(3), zeros(3) # Used to store terms of shoelace formula

    # Get the number of local modes retained for each subdomain
    for idom in 1:ndom
      push!(md, size(Φd[idom])[2])
    end

    # Loop over elements
    for iel in 1:nel
      
      # Get (x, y) coordinates of each element vertex
      for r in 1:3
        rnode = cells[r, iel]
        x[r], y[r] = points[1, rnode], points[2, rnode]
      end
      
      # Terms of the shoelace formula for a triangle
      Δx[1] = x[3] - x[2]
      Δx[2] = x[1] - x[3]
      Δx[3] = x[2] - x[1]
      Δy[1] = y[2] - y[3]
      Δy[2] = y[3] - y[1]
      Δy[3] = y[1] - y[2]
      
      # Area of element
      Area = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.

      # Get host subdomain
      idom = epart[iel]

      # Loop over local modes of subdomain
      for α in 1:md[idom]
        idom == 1 ? ind_dα = α : ind_dα = sum(md[1:idom-1]) + α

        # Loop over local modes of subdomain 
        for β in α:md[idom]
          idom == 1 ? ind_dβ = β : ind_dβ = sum(md[1:idom-1]) + β

          # Loop over vertices of element
          value = 0.
          for i in 1:3
            inode = inds_g2ld[idom][cells[i, iel]]
            ϕ_d_α_i = Φd[idom][inode, α]

            # Loop over vertices of element
            for j in 1:3
              jnode = inds_g2ld[idom][cells[j, iel]]
              ϕ_d_β_j = Φd[idom][jnode, β]
              if i == j
                value += ϕ_d_α_i * ϕ_d_β_j * Area / 6
              else
                value += ϕ_d_α_i * ϕ_d_β_j * Area / 12
              end
            end
          end

          push!(I, ind_dα)
          push!(J, ind_dβ)
          push!(V, value)
          if β != α
            push!(I, ind_dβ)
            push!(J, ind_dα)
            push!(V, value)
          end
        end
      end
    end
    
    # Assemble sparse mass matrix
    M = sparse(I, J, V)
    
    return M
  end


"""
do_global_mass_covariance_reduced_assembly(cells, points, cov)

Assembles Galerkin kernel operator generated by the covariance function
cov with 2D P1 finite elements with a given triangulation (cells, points).

Input:

cells[1:3, 1:nel]: Global node indices of local nodes in each element
cells[1, iel]: Global node index of 1st node of element iel ∈ [1, nel]
cells[2, iel]: Global node index of 2nd node of element iel ∈ [1, nel]
cells[3, iel]: Global node index of 3rd node of element iel ∈ [1, nel]

points[1:2, 1:nnode]: Coordinates of all nodal points
points[1, inode]: x-coordinate of global node indexed inode ∈ [1, nnode]
points[2, inode]: y-coordinate of global node indexed inode ∈ [1, nnode]

cov: function(x1::Float64, y1::Float64, x2::Float64, y2::Float64)::Float64 

Output:

C: Array of Galerkin formulation (nnode-by-nnode)
   with components C_ij = ∫_Ω ϕ_i(P) ∫_{Ω'} cov(P, P') ϕ_j(P') dΩ' dΩ.

# Examples
```jldoctest
julia>
using TriangleMesh
using Fem

poly = polygon_Lshape()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

function cov(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
  L = .1
  return exp(-((x1 - x2)^ 2 + (y1 - y2)^2) / L^2)
end

# Assembly for 11_894 DoFs
C = @time do_mass_covariance_assembly(mesh.cell, mesh.point, cov)

Maximum triangle area: .00005
 90.211575 seconds (243.67 k allocations: 2.121 GiB, 0.01% gc time)

```
"""
function do_global_mass_covariance_reduced_assembly(cells, points, elemsd, inds_g2ld, inds_l2gd, Φd, cov)
  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of nodes
  ndom = length(Φd) # Number of subdomains
  md = Int[] # Number of local modes for each subdomain
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3) # Used to store terms of shoelace formula
  Area = zeros(nel) # Used to store element areas

  # Get the number of local modes retained for each subdomain
  for idom in 1:ndom
    push!(md, size(Φd[idom])[2])
  end

  # Global reduced mass covariance matrix
  md_sum = sum(md)
  K = zeros(md_sum, md_sum)
  
  # Loop over subdomains
  for idom in 1:ndom
    nnode_idom = inds_g2ld[idom].count

    # Loop over subdomains
    for jdom in idom:ndom
      nnode_jdom = inds_g2ld[jdom].count
      
      R = zeros(nnode_idom, nnode_jdom) # R[i, j] ≈ ∑_e ∫_{Ω'_e} ϕ_i(P') cov(P', P_j) dΩ'
      C = zeros(nnode_idom, nnode_jdom) # C[i, j] ≈ ∫_Ω ϕ_i(P) ∫_{Ω'} cov(P, P') ϕ_j(P') dΩ' dΩ.

      # Loop over mesh nodes of the jdom-th subdomain 
      for (j, jnode) in enumerate(inds_l2gd[jdom])

        # Get coordinates of node
        xj = points[1, jnode]
        yj = points[2, jnode]
        
        # Loop over elements of the idom-th subdomain
        for (iel, el) in enumerate(elemsd[idom])
        
          # Get (x, y) coordinates of each element vertex
          for r in 1:3
            rnode = cells[r, el]
            x[r], y[r] = points[1, rnode], points[2, rnode]
          end
          
          # Terms of the shoelace formula for a triangle
          Δx[1] = x[3] - x[2]
          Δx[2] = x[1] - x[3]
          Δx[3] = x[2] - x[1]
          Δy[1] = y[2] - y[3]
          Δy[2] = y[3] - y[1]
          Δy[3] = y[1] - y[2]
          
          # Area of element
          Area_el = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
          
          # Add local contributions
          for r in 1:3
            s = r + 1 - floor(Int, (r + 1) / 3) * 3
            s == 0 ? s = 3 : nothing
            t = r + 2 - floor(Int, (r + 2) / 3) * 3
            t == 0 ? t = 3 : nothing
            i = inds_g2ld[idom][cells[r, el]]
            R[i, j] += (2 * cov(x[r], y[r], xj, yj) 
                          + cov(x[s], y[s], xj, yj) 
                          + cov(x[t], y[t], xj, yj)) * Area_el / 12
          end
        end # for el
      end # for jnode
      
      # Loop over mesh nodes of the idom-th subdomain 
      for (i, inode) in enumerate(inds_l2gd[idom])
        
        # Loop over elements of the jdom-th subdomain
        for (jel, el) in enumerate(elemsd[jdom])
        
          # Get (x, y) coordinates of each element vertex
          for r in 1:3
            rnode = cells[r, el]
            x[r], y[r] = points[1, rnode], points[2, rnode]
          end
          
          # Terms of the shoelace formula for a triangle
          Δx[1] = x[3] - x[2]
          Δx[2] = x[1] - x[3]
          Δx[3] = x[2] - x[1]
          Δy[1] = y[2] - y[3]
          Δy[2] = y[3] - y[1]
          Δy[3] = y[1] - y[2]
          
          # Area of element
          Area_el = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.

          # Add local contributions
          for r in 1:3
            s = r + 1 - floor(Int, (r + 1) / 3) * 3
            s == 0 ? s = 3 : nothing
            t = r + 2 - floor(Int, (r + 2) / 3) * 3
            t == 0 ? t = 3 : nothing
            j = inds_g2ld[jdom][cells[r, el]]
            k = inds_g2ld[jdom][cells[s, el]]
            ℓ = inds_g2ld[jdom][cells[t, el]]
            C[i, j] += (2 * R[i, j] 
                          + R[i, k]  
                          + R[i, ℓ]) * Area_el / 12
          end
        end # for el
      end # for inode

      # Loop over local modes of the idom-th subdomain
      for α in 1:md[idom]
        idom == 1 ? ind_α_idom = α : ind_α_idom = sum(md[1:idom-1]) + α 

        # Loop over local modes of the jdom-th subdomain
        for β in 1:md[jdom]
          jdom == 1 ? ind_β_jdom = β : ind_β_jdom = sum(md[1:jdom-1]) + β
          
          for i in 1:nnode_idom
            for j in 1:nnode_jdom
              K[ind_α_idom, ind_β_jdom] += Φd[idom][i, α] * Φd[jdom][j, β] * C[i, j]
            end
          end
          
          idom == jdom ? nothing : K[ind_β_jdom, ind_α_idom] = K[ind_α_idom, ind_β_jdom]

        end # end β
      end # end α
    end # for jdom
  end # for idom

  return K
end




function draw(mesh, Λ, Φ, ϕd, inds_l2gd)
    nnode = mesh.n_point # Number of mesh nodes
    nmode = length(Λ) # Number of global modes
    ndom = length(ϕd) # Number of subdomains
    md = Int[] # Number of local modes for each subdomain
  
    ξ = rand(Normal(), nmode)
    g = zeros(nnode)
    cnt = zeros(Int, nnode)
  
    # Get the number of local modes retained for each subdomain
    for idom in 1:ndom
      push!(md, size(ϕd[idom])[2])
    end
  
    for idom in 1:ndom
      for (i, inode) in enumerate(inds_l2gd[idom])
        cnt[inode] += 1
        for α in 1:md[idom]
          idom == 1 ? ind_α_idom = α : ind_α_idom = sum(md[1:idom-1]) + α
          for γ in 1:nmode
            g[inode] += sqrt(Λ[γ]) * ξ[γ] * Φ[ind_α_idom, γ] * ϕd[idom][i, α]
          end
        end # for α
      end # for γ
    end # for idom
  
    return ξ, g
  end
