using SparseArrays

"""
apply_dirichlet(e, p, A_mat, b_vec, uexact)

Applies Dirichlet boundary condition.

Input:

e[1:2, nbndnode]: Nodal points on boundary
e[1, ibndnode]: x-coordinate of boundary node indexed ibndnode ∈ [1, nbndnode]
e[2, ibndnode]: y-coordinate of boundary node indexed ibndnode ∈ [1, nbndnode]

p[1:2, 1:nnode]: coordinates of nodal points (not including at boundary?)
p[1, inode]: x-coordinate of node indexed inode ∈ [1, nnode]
p[2, inode]: y-coordinate of node indexed inode ∈ [1, nnode]

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
function apply_dirichlet(e, p, A_mat, b_vec, uexact)
  _, nnode = size(p) # Number of nodes
  _, npres = size(e) # Number of boundary points
  g1 = zeros(npres)
  
  # Evaluate solution at boundary points
  for i in 1:npres
    xb = p[1, e[1, i]]
    yb = p[2, e[1, i]]
    g1[i] = uexact(xb, yb)
  end
  
  # Loop of boundary points
  for i in 1:npres
    nod = e[1, i]
    
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