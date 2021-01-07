using SparseArrays

"""
get_dirichlet_inds(points::Array{Float64,2}, point_marker::Array{Int,2})

Get global to local indices maps for nodes in the boundary with Dirichlet
boundary condition, and .

Input:

points[1:2, 1:nnode]: coordinates of all nodal points
points[1, inode]: x-coordinate of global node indexed inode ∈ [1, nnode]
points[2, inode]: y-coordinate of global node indexed inode ∈ [1, nnode]

point_marker[inode]: coordinates of all nodal points

Output:

dirichlet_inds_g2l: 

not_dirichlet_inds_g2l:

"""
function get_dirichlet_inds(points::Array{Float64,2},
                            point_marker::Array{Int,2})
  _, nnode = size(points)

  dirichlet_inds_g2l = Dict{Int,Int}()
  not_dirichlet_inds_g2l = Dict{Int,Int}()

  dirichlet_inds_l2g = Int[]
  not_dirichlet_inds_l2g = Int[]

  for inode in 1:nnode
    if point_marker[inode] == 1
      dirichlet_inds_g2l[inode] = length(dirichlet_inds_g2l) + 1
      push!(dirichlet_inds_l2g, inode)
    else
      not_dirichlet_inds_g2l[inode] = length(not_dirichlet_inds_g2l) + 1
      push!(not_dirichlet_inds_l2g, inode)
    end
  end
  
  return dirichlet_inds_g2l, not_dirichlet_inds_g2l,
         dirichlet_inds_l2g, not_dirichlet_inds_l2g 
end


function append_bc()
  return 
end


"""
apply_dirichlet(segments, points, A_mat, b_vec, uexact)

Applies Dirichlet boundary condition.

Input:

segments[1:2, nbndseg]: Global node indices of boundary segments
segments[1, iseg]: node index at begining of segment iseg ∈ [1, nbndseg]
segments[2, iseg]: node index at end of segment iseg ∈ [1, nbndseg]

points[1:2, 1:nnode]: coordinates of all nodal points
points[1, inode]: x-coordinate of global node indexed inode ∈ [1, nnode]
points[2, inode]: y-coordinate of global node indexed inode ∈ [1, nnode]

A_mat::SparseMatrixCSC{Float64}

b_vec::Vector{Float64}

uexact: function(x::Float64, y::Float64)::Float64 > 0 ∀ x, y

Output:

A_mat: sparse array of Galerkin formulation (nnode-by-nnode)
       with Dirichlet boundary condition applied.

b_vec: right hand side vector of Galerkin formulation (nnode-by-1)
       with Dirichlet boundary condition applied.

# Examples
```jldoctest
julia>
using TriangleMesh
using Fem

poly = polygon_Lshape();
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true);

function a(x::Float64, y::Float64)
  return 1.
end

function f(x::Float64, y::Float64)
  return -1.
end

function uexact(xx, yy)
    return .734
end

# Assembly for 1_165_446 DoFs
A, b = @time do_isotropic_elliptic_assembly(mesh.cell, mesh.point);

# Apply Dirichlet boundary conditions
A, b = apply_dirichlet(e, mesh.point, A, b, uexact)

```
"""
function apply_dirichlet(segments, points, A_mat, b_vec, uexact)
  _, nnode = size(points) # Number of nodes
  _, nbndseg = size(segments) # Number of boundary segments
  g1 = zeros(nbndseg) # 
  
  # Evaluate solution at starting node of segments
  for i in 1:nbndseg
    xb = points[1, segments[1, i]]
    yb = points[2, segments[1, i]]
    g1[i] = uexact(xb, yb)
  end
  
  # Loop over boundary segments
  for i in 1:nbndseg
    nod = segments[1, i]
    
    # Loop over mesh nodes
    for k in 1:nnode
      b_vec[k] -= A_mat[k, nod] * g1[i]
      A_mat[nod, k] = 0
      A_mat[k, nod] = 0
    end
    
    # Change coefficients of related DoFs
    A_mat[nod, nod] = 1
    b_vec[nod] = g1[i]
  end
  
  return A_mat, b_vec
end