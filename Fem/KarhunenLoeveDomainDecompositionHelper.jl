

"""
suggest_parameters()
  
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
  
# Examples
```jldoctest
julia>
using TriangleMesh
using NPZ
using Fem

poly = polygon_unitSquare()
mesh = create_mesh(poly, info_str="my mesh", voronoi=true, delaunay=true, set_area_max=true)

ndom = 300
```
"""
function suggest_parameters(nnode::Int)
  if nnode >= 1_000_000
    return .99996, .99995
  elseif nnode >= 100_000
    return .9996, .9995
  else
    return .9996, .995
  end
  return relative_local, relative_global
end


function get_root_filename(model::String,
                           sig2::Float64,
                           L::Float64,
                           nnode::Int)
  fname = model * "_"
  fname *= "sig2$sig2" * "_"
  fname *= "L$L" * "_"
  return fname * "DoF$(nnode)_bis"
end

