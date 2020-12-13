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

# Triangulation data: p, t, e.
#
# nnodes: Number of nodes in mesh 
#
# p[1:2, 1:nnode]: coordinates of nodal points (not including at boundary?)
# p[1, inode]: x-coordinate of node indexed inode ∈ [1, nnode]
# p[2, inode]: y-coordinate of node indexed inode ∈ [1, nnode]
#
# t[1:3, 1:nel]: Node indices of elements
# t[1, iel]: 1st node index of element iel ∈ [1, nel]
# t[2, iel]: 2nd node index of element iel ∈ [1, nel]
# t[3, iel]: 3rd node index of element iel ∈ [1, nel]
#
# e[1:2, nbndnode]: Nodal points on boundary
# e[1, ibndnode]: x-coordinate of boundary node indexed ibndnode ∈ [1, nbndnode]
# e[2, ibndnode]: y-coordinate of boundary node indexed ibndnode ∈ [1, nbndnode]

function a(x, y)
  return 1.
end

function f(x, y)
  return -1.
end

function uexact(xx, yy)
    return .734
end


"""
do_assembly(nodes, p)

Does assembly of sparse Galerkin operator for 2D P1 finite elements with a 
given triangulation (nodes, p).

Input:

nodes[1:3, 1:nel]: Node indices of elements
nodes[1, iel]: 1st node index of element iel ∈ [1, nel]
nodes[2, iel]: 2nd node index of element iel ∈ [1, nel]
nodes[3, iel]: 3rd node index of element iel ∈ [1, nel]

p[1:2, 1:nnode]: coordinates of nodal points (not including at boundary?)
p[1, inode]: x-coordinate of node indexed inode ∈ [1, nnode]
p[2, inode]: y-coordinate of node indexed inode ∈ [1, nnode]

Output:

A_mat: sparse array of Galerkin formulation (nnode-by-nnode)
       with components Amat_ij = ∫_Ω a ∇ϕ_i ⋅ ∇ϕ_j dΩ where 
       a: Ω → R is also projected on Span{ϕ_k}_{k=1}^nnode.

b_vec: right hand side vector of Galerkin formulation (nnode-by-1)

# Examples
```jldoctest
julia>
using TriangleMesh;
push!(LOAD_PATH, ".");
using Fem;

poly = polygon_Lshape();
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true);

# Assembly for 1_165_446 DoFs
A, b = @time do_assembly(mesh.cell, mesh.point);

Maximum triangle area: .0000005
  2.584544 seconds (685.03 k allocations: 1.228 GiB, 10.03% gc time)

```
"""
function do_assembly(nodes, p)
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
    coeff = 0.
    for j in 1:3
      jj = nodes[j, iel]
      x[j], y[j] = p[1, jj], p[2, jj]
      coeff += a(x[j], y[j])
    end
    coeff /= 3.
    #
    # Terms of shoelace formula for a triangle
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
      m = i + 2 - floor(Int, (i + 2) / 3) * 3
      m == 0 ? m = 3 : nothing
      #
      ii = nodes[i, iel]
      b_vec[ii] += (2 * f(x[i], y[i]) + f(x[j], y[j]) + f(x[m], y[m])) / Area / 12
    end
  end
  A_mat = sparse(I, J, V)
  #
  return A_mat, b_vec
end




"""
apply_dirichlet(e, p, nnode, gk, gf, g1)

Applies Dirichlet boundary condition.

```
"""
function apply_dirichlet(e, p, nnode, A_mat, b_vec, g1)
  _, npres = size(e)
  g1 = zeros(npres)
  for i in 1:npres
    xb = p[1, e[1, i]]
    yb = p[2, e[1, i]]
    g1[i] = uexact(xb, yb)
  end
  for i in 1:npres
    nod = e[1, i]
    for k in 1:nnode
      b_vec[k] -= A_mat[k, nod] * g1[i]
      A_mat[nod, k] = 0
      A_mat[k, nod] = 0
    end
    A_mat[nod, nod] = 1
    b_vec[nod] = g1[i]
  end
  return A_mat, b_vec
end