using TriangleMesh
using SparseArrays
using PyPlot

function plot_TriMesh(m :: TriMesh; 
                        linewidth :: Real = 1, 
                        marker :: String = "None",
                        markersize :: Real = 10,
                        linestyle :: String = "-",
                        color :: String = "red")

    fig = matplotlib[:pyplot][:figure]("2D Mesh Plot", figsize = (10,10))
    
    ax = matplotlib[:pyplot][:axes]()
    ax[:set_aspect]("equal")
    
    # Connectivity list -1 for Python
    tri = ax[:triplot](m.point[1,:], m.point[2,:], m.cell'.-1 )
    setp(tri,   linestyle = linestyle,
                linewidth = linewidth,
                marker = marker,
                markersize = markersize,
                color = color)
    
    fig[:canvas][:draw]()
    println("yrdy")    
    return fig
end


"""
do_isotropic_elliptic_assembly(nodes, p, a, f)

Does assembly of sparse Galerkin operator and right hand side 
for 2D P1 finite elements with a given triangulation (nodes, p).

Input:

nodes[1:3, 1:nel]: Node indices of elements
nodes[1, iel]: 1st node index of element iel ∈ [1, nel]
nodes[2, iel]: 2nd node index of element iel ∈ [1, nel]
nodes[3, iel]: 3rd node index of element iel ∈ [1, nel]

p[1:2, 1:nnode]: coordinates of nodal points (not including at boundary?)
p[1, inode]: x-coordinate of node indexed inode ∈ [1, nnode]
p[2, inode]: y-coordinate of node indexed inode ∈ [1, nnode]

a: function(x::Float64, y::Float64)::Float64 > 0 ∀ x, y

f: function(x::Float64, y::Float64)::Float64

Output:

A_mat: sparse array of Galerkin formulation (nnode-by-nnode)
       with components A_mat_ij = ∫_Ω a ∇ϕ_i ⋅ ∇ϕ_j dΩ where 
       a: Ω → R is interpolated at the nodes in Span {ϕ_k}_{k=1}^nnode.

b_vec: right hand side vector of Galerkin formulation (nnode-by-1)
       with components b_vec_i = ∫_Ω f ϕ_i dΩ where f: Ω → R is 
       interpolated at the nodes in Span {ϕ_k}_{k=1}^nnode.

# Examples
```jldoctest
julia>
using TriangleMesh
using Fem

poly = polygon_Lshape()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

function a(x::Float64, y::Float64)
  return 1. + x * y
end

function f(x::Float64, y::Float64)
  return -1.
end

# Assembly for 1_165_446 DoFs
A, b = @time do_isotropic_elliptic_assembly(mesh.cell, mesh.point)

Maximum triangle area: .0000005
  2.584544 seconds (685.03 k allocations: 1.228 GiB, 10.03% gc time)

```
"""
function do_isotropic_elliptic_assembly(nodes, p, a, f)
  _, nel = size(nodes) # Number of elements
  _, nnode = size(p) # Number of nodes
  I, J, V = Int[], Int[], Float64[] # Indices (I, J) and data (V) for sparse Galerkin operator
  b_vec = zeros(nnode, 1) # Right hand side
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3)
  #
  # Loop over elements
  for iel in 1:nel
    #
    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    coeff = 0.
    for j in 1:3
      jj = nodes[j, iel]
      x[j], y[j] = p[1, jj], p[2, jj]
      coeff += a(x[j], y[j])
    end
    coeff /= 3.
    #
    # Terms of the shoelace formula for a triangle
    Δx[1] = x[3] - x[2]
    Δx[2] = x[1] - x[3]
    Δx[3] = x[2] - x[1]
    Δy[1] = y[2] - y[3]
    Δy[2] = y[3] - y[1]
    Δy[3] = y[1] - y[2]
      #
      # Area of element
      Area = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
    #
    # Add local stiffness contributions from element
    for i in 1:3
      ii = nodes[i, iel]
      for j in 1:3
        Kij = coeff * (Δy[i] * Δy[j] + Δx[i] * Δx[j]) / 4 / Area
        jj = nodes[j, iel]
        push!(I, ii)
        push!(J, jj)
        push!(V, Kij)
      end
    end
    #
    # Add right hand side contributions from element
    for i in 1:3
      j = i + 1 - floor(Int, (i + 1) / 3) * 3
      j == 0 ? j = 3 : nothing
      k = i + 2 - floor(Int, (i + 2) / 3) * 3
      k == 0 ? k = 3 : nothing
      #
      ii = nodes[i, iel]
      b_vec[ii] += (2 * f(x[i], y[i]) + f(x[j], y[j]) + f(x[k], y[k])) * Area / 12
    end
  end
  #
  # Assemble sparse array of Galerkin operator
  A_mat = sparse(I, J, V)
  #
  return A_mat, b_vec
end


"""
do_covariance_mass_assembly(nodes, p, cov)

Does assembly of sparse Galerkin operator and right hand side 
for 2D P1 finite elements with a given triangulation (nodes, p).

Input:

nodes[1:3, 1:nel]: Node indices of elements
nodes[1, iel]: 1st node index of element iel ∈ [1, nel]
nodes[2, iel]: 2nd node index of element iel ∈ [1, nel]
nodes[3, iel]: 3rd node index of element iel ∈ [1, nel]

p[1:2, 1:nnode]: coordinates of nodal points (not including at boundary?)
p[1, inode]: x-coordinate of node indexed inode ∈ [1, nnode]
p[2, inode]: y-coordinate of node indexed inode ∈ [1, nnode]

cov: function(x1::Float64, y1::Float64, x2::Float64, y2::Float64)::Float64 

Output:

C: Array of Galerkin formulation (nnode-by-nnode)
   with components A_mat_ij = ∫_Ω a ∇ϕ_i ⋅ ∇ϕ_j dΩ where 
   a: Ω → R is interpolated at the nodes in Span {ϕ_k}_{k=1}^nnode.

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
function do_mass_covariance_assembly(nodes, p, cov)
  _, nel = size(nodes) # Number of elements
  _, nnode = size(p) # Number of nodes
  I, J, V = Int[], Int[], Float64[] # Indices (I, J) and data (V) for sparse Galerkin operator
  b_vec = zeros(nnode, 1) # Right hand side
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3)
  R = zeros(nnode, nnode)
  C = zeros(nnode, nnode)
  Area = zeros(nel)
  #
  # Loop over nodes
  for j in 1:nnode
    #
    # Get coordinates of node
    xj = p[1, j]
    yj = p[2, j]
    #
    # Loop over elements
    for iel in 1:nel
      #
      # Get (x, y) coordinates of each element vertex
      # and coefficient at the center of the element
      for r in 1:3
        rr = nodes[r, iel]
        x[r], y[r] = p[1, rr], p[2, rr]
      end
      #
      # Terms of the shoelace formula for a triangle
      Δx[1] = x[3] - x[2]
      Δx[2] = x[1] - x[3]
      Δx[3] = x[2] - x[1]
      Δy[1] = y[2] - y[3]
      Δy[2] = y[3] - y[1]
      Δy[3] = y[1] - y[2]
      #
      # Area of element
      Area_iel = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
      Area[iel] = Area_iel
      #
      # Add local contributions
      for r in 1:3
        s = r + 1 - floor(Int, (r + 1) / 3) * 3
        s == 0 ? s = 3 : nothing
        t = r + 2 - floor(Int, (r + 2) / 3) * 3
        t == 0 ? t = 3 : nothing
        i = nodes[r, iel]
        R[i, j] += (2 * cov(x[r], y[r], xj, yj) 
                      + cov(x[s], y[s], xj, yj) 
                      + cov(x[t], y[t], xj, yj)) * Area_iel / 12
      end
    end
  end
  #
  # Loop over nodes
  for i in 1:nnode
    #
    # Loop over elements
    for iel in 1:nel
      #
      # Get area of element
      Area_iel = Area[iel]
      #
      # Add local contributions
      for r in 1:3
        s = r + 1 - floor(Int, (r + 1) / 3) * 3
        s == 0 ? s = 3 : nothing
        t = r + 2 - floor(Int, (r + 2) / 3) * 3
        t == 0 ? t = 3 : nothing
        j = nodes[r, iel]
        k = nodes[s, iel]
        ℓ = nodes[t, iel]
        C[i, j] += (2 * R[i, j] 
                      + R[i, k]  
                      + R[i, ℓ]) * Area_iel / 12
      end
    end
  end
  #
  return C
end


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
using TriangleMesh;
push!(LOAD_PATH, ".");
using Fem;

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
  #
  # Evaluate solution at boundary points
  for i in 1:npres
    xb = p[1, e[1, i]]
    yb = p[2, e[1, i]]
    g1[i] = uexact(xb, yb)
  end
  #
  # Loop of boundary points
  for i in 1:npres
    nod = e[1, i]
    #
    # Loop over mesh nodes
    for k in 1:nnode
      b_vec[k] -= A_mat[k, nod] * g1[i]
      A_mat[nod, k] = 0
      A_mat[k, nod] = 0
    end
    #
    A_mat[nod, nod] = 1
    b_vec[nod] = g1[i]
  end
  #
  return A_mat, b_vec
end