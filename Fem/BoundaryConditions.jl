"""
get_dirichlet_inds(points::Array{Float64,2}, 
                   point_marker::Array{Int,2})

Get global to local indices maps for nodes in the boundary with Dirichlet
boundary condition, and .

Input:

 `points::Array{Float64,2}`, `size(cells) = (2, nnode)`,
 (x,y)-coordinates of all mesh nodes.

 `point_marker::Array{Int,1}`, `size(point_marker) = (nnode,)`,
  indicates if a node is Dirichlet (1) or not (0).

Output:

 `dirichlet_inds_g2l::Dict{Int,Int}`,
  conversion table from global mesh node indices to indices
  for Dirichlet nodes. 

 `not_dirichlet_inds_g2l::Dict{Int,Int}`,
  conversion table from global mesh node indices to indices
  for non-Dirichlet nodes. 

 `dirichlet_inds_l2g::Array{Int,1}`,
  conversion table from local indices of Dirichlet nodes to global 
  mesh node indices.

 `not_dirichlet_inds_l2g::Array{Int,1}`, 
  conversion table from local indices of non-Dirichlet nodes to global 
  mesh node indices.

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


"""
append_bc(dirichlet_inds_l2g::Array{Int,1},
          not_dirichlet_inds_l2g::Array{Int,1},
          u_no_dirichlet::Array{Float64,1},
          points::Array{Float64,2},
          uexact::Function)

Appends solution to Dirichlet nodes

Input:

 `dirichlet_inds_l2g::Array{Int,1}`,
  conversion table from local indices of Dirichlet nodes to global 
  mesh node indices.

 `not_dirichlet_inds_l2g::Array{Int,1}`, 
  conversion table from local indices of non-Dirichlet nodes to global 
  mesh node indices.

 `u_no_dirichlet::Array{Float64,1}`,
  solution at non-Dirichlet mesh nodes.

 `points::Array{Float64,2}`, `size(cells) = (2, nnode)`,
 (x,y)-coordinates of all mesh nodes.

 `uexact::Function`,
  prescribed u.

Output:

 `u::Array{Float64,1}`, `size(u) = (nnode,)`,
  solution at all the mesh nodes.

"""
function append_bc(dirichlet_inds_l2g::Array{Int,1},
                   not_dirichlet_inds_l2g::Array{Int,1},
                   u_no_dirichlet::Array{Float64,1},
                   points::Array{Float64,2},
                   uexact::Function)
  
  nnode_dirichlet = length(dirichlet_inds_l2g)
  nnode_not_dirichlet = length(not_dirichlet_inds_l2g)

  u = Array{Float64,1}(undef, nnode_dirichlet +
                              nnode_not_dirichlet)

  for (i, ind) in enumerate(dirichlet_inds_l2g)
    u[ind] = uexact(points[1, ind], points[2, ind])
  end

  for (i, ind) in enumerate(not_dirichlet_inds_l2g)
    u[ind] = u_no_dirichlet[i] 
  end

  return u
end


"""
apply_dirichlet(segments, points, A_mat, b_vec, uexact)

Applies Dirichlet boundary condition.

Input:

 `segments::Array{Int,2}`, `size(segments) = (2, nbndseg)`,
  pairs of global node indices of each boundary segment.

 `points::Array{Float64,2}`, `size(cells) = (2, nnode)`,
 (x,y)-coordinates of all mesh nodes.

 `A_mat::SparseMatrixCSC{Float64}`, `size(A_mat) = (nnode, nnode)`,
  sparse array of Galerkin formulation before application Dirichlet boundary conditions.

 `b_vec::Array{Float64,1}`, `size(b_vec) = (nnode,)`,
  right hand side vector of Galerkin formulation before application of Dirichlet 
  boundary conditions.

 `uexact::Function`, `uexact(x::Float64, y::Float64)::Float64`,
  prescribed u.

Output:

 `A_mat::SparseMatrixCSC{Float64}`, `size(A_mat) = (nnode, nnode)`,
  sparse array of Galerkin formulation with Dirichlet boundary condition applied.

 `b_vec::Array{Float64,1}`, `size(b_vec) = (nnode,)`
  right hand side vector of Galerkin formulation with Dirichlet 
  boundary condition applied.

"""
function apply_dirichlet(segments::Array{Int,2},
                         points::Array{Float64,2},
                         A_mat::SparseMatrixCSC{Float64},
                         b_vec::Array{Float64,1},
                         uexact::Function)

  _, nnode = size(points) # Number of nodes
  _, nbndseg = size(segments) # Number of boundary segments
  g1 = zeros(nbndseg)
  
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
