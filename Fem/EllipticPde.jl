"""
     function do_isotropic_elliptic_assembly(cells::Array{Int,2},
                                             points::Array{Float64,2},
                                             dirichlet_inds_g2l::Dict{Int,Int},
                                             not_dirichlet_inds_g2l::Dict{Int,Int},
                                             point_marker::Array{Int,2},
                                             a::Function,
                                             f::Function,
                                             u_exact::Function)

Does assembly of sparse Galerkin operator and right hand side 
for 2D P1 finite elements with a given triangulation (cells, points).

Input:

 `cells::Array{Int,2}`, `size(cells) = (3, n_el)`,
  global node indices of local nodes in each element.

 `points::Array{Float64,2}`, `size(points) = (2, nnode)`,
 (x,y)-coordinates of all nodal points.
 
 `a(x::Float64, y::Float64)::Float64` > `0` ∀ `(x, y)`,
 coefficient field function.

 `f(x::Float64, y::Float64)::Float64`,
  source term.

Output:

 `A_mat::SparseMatrixCSC{Float64,Int}`, `size(A_mat) = (n, n)`, 
  sparse array of Galerkin formulation (`n`-by-`n`) with components 
 `A_mat[i, j] = ∫_Ω a ∇ϕ_i ⋅ ∇ϕ_j` dΩ where `a: Ω → R` is interpolated 
  at the nodes in `Span {ϕ_k}_{k=1}^n`.

 `b_vec::Array{Float64,1}`, `size(b_vec) = (n,)`,
  right hand side vector of Galerkin formulation with with components 
 `b_vec[i] = ∫_Ω f ϕ_i dΩ` where `f: Ω → R` is interpolated at the nodes 
  in `Span {ϕ_k}_{k=1}^n`.

# Examples
```jldoctest
julia>
using Fem

tentative_nnode = 400_000
mesh = get_mesh(tentative_nnode)

function a(x::Float64, y::Float64)
  return 1. + x * y
end

function f(x::Float64, y::Float64)
  return -1.
end

A, b = do_isotropic_elliptic_assembly(mesh.cell, mesh.point)

```
"""
function do_isotropic_elliptic_assembly(cells::Array{Int,2},
                                        points::Array{Float64,2},
                                        dirichlet_inds_g2l::Dict{Int,Int},
                                        not_dirichlet_inds_g2l::Dict{Int,Int},
                                        point_marker::Array{Int,2},
                                        a::Function,
                                        f::Function,
                                        u_exact::Function)

  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of nodes
  I, J, V = Int[], Int[], Float64[] # Indices (I, J) and data (V) for sparse Galerkin operator
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3)
  
  b_vec = zeros(Float64, length(not_dirichlet_inds_g2l)) # Right hand side
  
  # Loop over elements
  for iel in 1:nel
  
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
      inode = cells[i, iel]
      
      i_is_dirichlet = point_marker[inode] == 1
      i_is_dirichlet ? inode = dirichlet_inds_g2l[inode] : inode = not_dirichlet_inds_g2l[inode]
    
      # Loop over vertices of element
      for j in 1:3
        # Store local contribution
        Kij = coeff * (Δy[i] * Δy[j] + Δx[i] * Δx[j]) / 4 / Area
        jnode = cells[j, iel]
        j_is_dirichlet = point_marker[jnode] == 1
        j_is_dirichlet ? jnode = dirichlet_inds_g2l[jnode] : jnode = not_dirichlet_inds_g2l[jnode]

        # Add stiffness contribution
        if !i_is_dirichlet & !j_is_dirichlet
          push!(I, inode)
          push!(J, jnode)
          push!(V, Kij)

        # Correct right hand side
        elseif i_is_dirichlet & !j_is_dirichlet
          b_vec[jnode] -= u_exact(x[i], y[i]) * Kij 
        end
      end # for j
    end # for i

    # Add right hand side contributions from element
    for i in 1:3
      j = i + 1 - floor(Int, (i + 1) / 3) * 3
      j == 0 ? j = 3 : nothing
      k = i + 2 - floor(Int, (i + 2) / 3) * 3
      k == 0 ? k = 3 : nothing
      inode = cells[i, iel]
      if point_marker[inode] == 0
        inode = not_dirichlet_inds_g2l[inode]
        b_vec[inode] += (2 * f(x[i], y[i]) 
                           + f(x[j], y[j])
                           + f(x[k], y[k])) * Area / 12
      end
    end 
  end # for iel

  # Assemble sparse array of Galerkin operator
  A_mat = sparse(I, J, V)

  return A_mat, b_vec
end


"""
     function do_isotropic_elliptic_assembly(cells::Array{Int,2},
                                             points::Array{Float64,2},
                                             point_marker::Array{Int,2},
                                             a::Function,
                                             f::Function,
                                             uexact::Function)

"""
function do_isotropic_elliptic_assembly(cells::Array{Int,2},
                                        points::Array{Float64,2},
                                        point_marker::Array{Int,2},
                                        a::Function,
                                        f::Function,
                                        uexact::Function)

  dirichlet_inds_g2l, not_dirichlet_inds_g2l = get_dirichlet_inds(points, point_marker)

  do_isotropic_elliptic_assembly(cells, points, dirichlet_inds_g2l,
                                 not_dirichlet_inds_g2l, 
                                 point_marker, a, f, uexact)
end


"""
     function do_isotropic_elliptic_assembly(cells::Array{Int,2},
                                             points::Array{Float64,2},
                                             dirichlet_inds_g2l::Dict{Int,Int},
                                             not_dirichlet_inds_g2l::Dict{Int,Int},
                                             point_marker::Array{Int,2},
                                             coeff::Array{Float64,1},
                                             f::Function,
                                             u_exact::Function)

"""
function do_isotropic_elliptic_assembly(cells::Array{Int,2},
                                        points::Array{Float64,2},
                                        dirichlet_inds_g2l::Dict{Int,Int},
                                        not_dirichlet_inds_g2l::Dict{Int,Int},
                                        point_marker::Array{Int,2},
                                        coeff::Array{Float64,1},
                                        f::Function,
                                        u_exact::Function)

  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of nodes
  I, J, V = Int[], Int[], Float64[] # Indices (I, J) and data (V) for sparse Galerkin operator
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3)

  b_vec = zeros(Float64, length(not_dirichlet_inds_g2l)) # Right hand side

  # Loop over elements
  for iel in 1:nel

    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    Δa = 0.
    for (j, jnode) in enumerate(cells[:, iel])
      x[j], y[j] = points[1, jnode], points[2, jnode]
      Δa += coeff[jnode]
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

    # Loop over vertices of element
    for i in 1:3
      inode = cells[i, iel]

      i_is_dirichlet = point_marker[inode] == 1
      i_is_dirichlet ? inode = dirichlet_inds_g2l[inode] : inode = not_dirichlet_inds_g2l[inode]

      # Loop over vertices of element
      for j in 1:3
        # Store local contribution
        Kij = Δa * (Δy[i] * Δy[j] + Δx[i] * Δx[j]) / 4 / Area
        jnode = cells[j, iel]
        j_is_dirichlet = point_marker[jnode] == 1
        j_is_dirichlet ? jnode = dirichlet_inds_g2l[jnode] : jnode = not_dirichlet_inds_g2l[jnode]

        # Add stiffness contribution
        if !i_is_dirichlet & !j_is_dirichlet
          push!(I, inode)
          push!(J, jnode)
          push!(V, Kij)

          # Correct right hand side
        elseif i_is_dirichlet & !j_is_dirichlet
          b_vec[jnode] -= u_exact(x[i], y[i]) * Kij 
        end
      end # for j
    end # for i

    # Add right hand side contributions from element
    for i in 1:3
      j = i + 1 - floor(Int, (i + 1) / 3) * 3
      j == 0 ? j = 3 : nothing
      k = i + 2 - floor(Int, (i + 2) / 3) * 3
      k == 0 ? k = 3 : nothing
      inode = cells[i, iel]
      if point_marker[inode] == 0
        inode = not_dirichlet_inds_g2l[inode]
        b_vec[inode] += (2 * f(x[i], y[i]) 
                           + f(x[j], y[j])
                           + f(x[k], y[k])) * Area / 12
      end
    end 
  end # for iel

  # Assemble sparse array of Galerkin operator
  A_mat = sparse(I, J, V)

  return A_mat, b_vec
end


"""
     function update_isotropic_elliptic_assembly!(A_mat::SparseMatrixCSC{Float64,Int},
                                                  b_vec::Array{Float64,1},
                                                  cells::Array{Int,2},
                                                  points::Array{Float64,2},
                                                  dirichlet_inds_g2l::Dict{Int,Int},
                                                  not_dirichlet_inds_g2l::Dict{Int,Int},
                                                  point_marker::Array{Int,2},
                                                  coeff::Array{Float64,1},
                                                  f::Function,
                                                  u_exact::Function)

"""
function update_isotropic_elliptic_assembly!(A_mat::SparseMatrixCSC{Float64,Int},
                                             b_vec::Array{Float64,1},
                                             cells::Array{Int,2},
                                             points::Array{Float64,2},
                                             dirichlet_inds_g2l::Dict{Int,Int},
                                             not_dirichlet_inds_g2l::Dict{Int,Int},
                                             point_marker::Array{Int,2},
                                             coeff::Array{Float64,1},
                                             f::Function,
                                             u_exact::Function)

  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of nodes
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3)

  A_mat .= 0
  b_vec .= 0

  # Loop over elements
  for iel in 1:nel

    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    Δa = 0.
    for (j, jnode) in enumerate(cells[:, iel])
      x[j], y[j] = points[1, jnode], points[2, jnode]
      Δa += coeff[jnode]
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

    # Loop over vertices of element
    for i in 1:3
      inode = cells[i, iel]

      i_is_dirichlet = point_marker[inode] == 1
      i_is_dirichlet ? inode = dirichlet_inds_g2l[inode] : inode = not_dirichlet_inds_g2l[inode]

      # Loop over vertices of element
      for j in 1:3
        # Store local contribution
        Kij = Δa * (Δy[i] * Δy[j] + Δx[i] * Δx[j]) / 4 / Area
        jnode = cells[j, iel]
        j_is_dirichlet = point_marker[jnode] == 1
        j_is_dirichlet ? jnode = dirichlet_inds_g2l[jnode] : jnode = not_dirichlet_inds_g2l[jnode]

        # Add stiffness contribution
        if !i_is_dirichlet & !j_is_dirichlet
          A_mat[inode, jnode] += Kij
          #push!(I, inode)
          #push!(J, jnode)
          #push!(V, Kij)

        # Correct right hand side
        elseif i_is_dirichlet & !j_is_dirichlet
          b_vec[jnode] -= u_exact(x[i], y[i]) * Kij 
        end
      end # for j
    end # for i

    # Add right hand side contributions from element
    for i in 1:3
      j = i + 1 - floor(Int, (i + 1) / 3) * 3
      j == 0 ? j = 3 : nothing
      k = i + 2 - floor(Int, (i + 2) / 3) * 3
      k == 0 ? k = 3 : nothing
      inode = cells[i, iel]
      if point_marker[inode] == 0
        inode = not_dirichlet_inds_g2l[inode]
        b_vec[inode] += (2 * f(x[i], y[i]) 
                           + f(x[j], y[j])
                           + f(x[k], y[k])) * Area / 12
      end
    end 
  end # for iel
end

  
"""
     function get_mass_matrix(cells::Array{Int,2}, points::Array{Float64,2})

Gets mass matrix of linear form using 2D P1 finite elements with a 
given triangulation (`cells`, `points`).

Input:

 `cells::Array{Int,2}`, `size(cells) = (3, n_el)`,
  global node indices of local nodes in each element.

 `points::Array{Float64,2}`, `size(points) = (2, nnode)`,
  (x, y)-coordinates of all nodal points.

Output:

 `M::SparseMatrixCSC{Float64,Int}`, `size(M) = (n, n)`, 
  masss matrix formulation (`n`-by-`n`) with components 
 `M[i, j] = ∫_Ω ϕ_i(P) ϕ_j(P) dΩ`.

# Examples
```jldoctest
julia>
using Fem

tentative_nnode = 400_000
mesh = get_mesh(tentative_nnode)

M = get_mass_matrix(mesh.cell, mesh.point)

```
"""
function get_mass_matrix(cells::Array{Int,2}, points::Array{Float64,2})

  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of cells
  I, J, V = Int[], Int[], Float64[] # Indices (I, J) and data (V) for mass matrix
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3) # Used to store terms of shoelace formula
  
  # Loop over elements
  for iel in 1:nel
    
    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    for r in 1:3
      rr = cells[r, iel]
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
    Area = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
    
    # Loop over vertices of element
    for i in 1:3
      ii = cells[i, iel]

      # Loop over vertices of element
      for j in 1:3
        
        # Store local contribution
        if i == j
          Kij = Area / 6.
        else
          Kij = Area / 12.
        end
        jj = cells[j, iel]
        push!(I, ii)
        push!(J, jj)
        push!(V, Kij)
      end
    end
  end
  
  # Assemble sparse mass matrix
  M = sparse(I, J, V)
  
  return M
end
