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


function f(xx, yy)
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

Amat: sparse array of Galerkin formulation (nnode-by-nnode)
bvec: right hand side vector of Galerkin formulation (nnode-by-1)

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
  bvec = zeros(nnode, 1) # Right hand side
  xx, yy = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  a, b, c = zeros(3), zeros(3), zeros(3) 
  #
  # Loop over elements
  for iel in 1:nel
    #
    # Get (x, y) coordinates of each element vertex
    for j in 1:3
      jj = nodes[j, iel]
      xx[j], yy[j] = p[1, jj], p[2, jj]
    end
    #
    # Compute area of element 
    for i in 1:3
      j = i + 1 - floor(Int, (i + 1) / 3) * 3
      j == 0 ? j = 3 : nothing
      m = i + 2 - floor(Int, (i + 2) / 3) * 3
      m == 0 ? m = 3 : nothing
      a[i] = xx[j] * yy[m] - xx[m] * yy[j]
      b[i] = yy[j] - yy[m]
      c[i] = xx[m] - xx[j]
    end
    Δel = (c[3] * b[2] - c[2] * b[3]) / 2.
    #
    # Add stiffness contributions from element
    for ir in 1:3
      ii = nodes[ir, iel]
      for ic in 1:3
        ak = (b[ir] * b[ic] + c[ir] * c[ic]) / 4 / Δel
        jj = nodes[ic, iel]
        push!(I, ii)
        push!(J, jj)
        push!(V, ak)
      end
    end
    #
    # Add right hand side contributions from element
    for ir in 1:3
      ii = nodes[ir, iel]
      j = ir + 1 - floor(Int, (ir + 1) / 3) * 3
      j == 0 ? j = 3 : nothing
      m = ir + 2 - floor(Int, (ir + 2) / 3) * 3
      m == 0 ? m = 3 : nothing
      bvec[ii] += (2 * f(xx[ir], yy[ir]) + f(xx[j], yy[j]) + f(xx[m], yy[m])) / Δel / 12
    end
  end
  Amat = sparse(I, J, V)
  #
  return Amat, bvec
end


"""
apply_dirichlet(e, p, nnode, gk, gf, g1)

Applies Dirichlet boundary condition.

```
"""
function apply_dirichlet(e, p, nnode, Amat, bvec, g1)
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
      bvec[k] -= Amat[k, nod] * g1[i]
      Amat[nod, k] = 0
      Amat[k, nod] = 0
    end
    Amat[nod, nod] = 1
    bvec[nod] = g1[i]
  end
  return Amat, bvec
end