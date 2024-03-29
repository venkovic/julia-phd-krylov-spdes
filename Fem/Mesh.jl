"""
     function get_mesh(tentative_nnode::Int;
                       keep_segments=true)

Generates and returns a triangular mesh of a 2D square with approximately
tentative_nnode mesh nodes.  

Input:

 `tentative_nnode::Int`,
  tentative number of DoFs for the triangular mesh.

 `keep_segments=true`

Output:

 `mesh::TriangleMesh.TriMesh`,
  triangular mesh.

"""
function get_mesh(tentative_nnode::Int;
                  keep_segments=true)
                  
  poly = TriangleMesh.polygon_unitSquare()
  mesh = TriangleMesh.create_mesh(poly, voronoi=true, delaunay=true)
  divide_cell_into = ceil(Int, .645 * tentative_nnode)
  mesh = TriangleMesh.refine(mesh, 
                             divide_cell_into=divide_cell_into,
                             keep_segments=keep_segments)
  return mesh
end


"""
     function save_mesh(mesh::TriangleMesh.TriMesh,
                        tentative_nnode::Int)

Saves mesh data in NPZ files.

Input:

 `mesh::TriangleMesh.TriMesh`,
  triangular mesh.

 `tentative_nnode::Int`,
  tentative number of DoFs for the triangular mesh.

"""
function save_mesh(mesh::TriangleMesh.TriMesh,
                   tentative_nnode::Int)

  npzwrite("data/DoF$tentative_nnode.cells.npz", mesh.cell' .- 1)
  npzwrite("data/DoF$tentative_nnode.points.npz", mesh.point')
  npzwrite("data/DoF$tentative_nnode.point_markers.npz", mesh.point_marker')
  npzwrite("data/DoF$tentative_nnode.cell_neighbors.npz", mesh.cell_neighbor')
end


"""
     function load_mesh(tentative_nnode::Int)

Loads mesh data from NPZ files.

Input:

 `tentative_nnode::Int`,
  tentative number of DoFs for the triangular mesh.

Output:

 `cells::Array{Int,2}`, `size(cells) = (3, n_el)`,
  nodes of each element.

 `points::Array{Float64,2}`, `size(cells) = (2, nnode)`,
  (x,y)-coordinates of DoFs.

 `point_markers::Array{Int,1}`, `size(point_markers) = (nnode,)`,
  indicates whether a node in Dirichlet (1) or not (0).

 `cell_neighbors::Array{Int,2}`, `size(cell_neighbors) = (3, n_el)`,
  neighboring elements of each element, or `-1` if edge is at mesh boundary. 

"""
function load_mesh(tentative_nnode::Int)

  cells = Array(npzread("data/DoF$tentative_nnode.cells.npz")') .+ 1
  points = Array(npzread("data/DoF$tentative_nnode.points.npz")')
  point_markers = Array(npzread("data/DoF$tentative_nnode.point_markers.npz")')
  cell_neighbors = Array(npzread("data/DoF$tentative_nnode.cell_neighbors.npz")')
  return cells, points, point_markers, cell_neighbors
end


"""
     function get_total_area(cells::Array{Int,2},
                             points::Array{Float64,2})

Gets total area of trianular mesh.

Input:

 `cells::Array{Int,2}`, `size(cells) = (3, n_el)`,
  nodes of each element.

 `points::Array{Float64,2}`, `size(points) = (2, nnode)`,
 (x,y)-coordinates of all mesh points.

Output:
 `area::Float64`,
  total area of mesh.

"""
function get_total_area(cells::Array{Int,2}, 
                        points::Array{Float64,2})

  _, nel = size(cells) # Number of elements
  area = 0. # Total mesh area
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3)

  # Loop over elements
  for iel in 1:nel

    # Get (x, y) coordinates of each element vertex
    # and coefficient at the center of the element
    for j in 1:3
      jj = cells[j, iel]
      x[j], y[j] = points[1, jj], points[2, jj]
    end
    
    # Terms of the shoelace formula for a triangle
    Δx[1] = x[3] - x[2]
    Δx[2] = x[1] - x[3]
    Δx[3] = x[2] - x[1]
    Δy[1] = y[2] - y[3]
    Δy[2] = y[3] - y[1]
    Δy[3] = y[1] - y[2]
    
    # Area of element
    area += (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
  end

  return area
end


"""
     function mesh_partition(cells::Array{Int,2}, ndom::Int)

Gets mesh partition using metis.

Input:

 `cells::Array{Int,2}`, `size(cells) = (3, n_el)`,
  nodes of each element.

 `ndom::Int`,
  number of subdomains.

Output:

 `epart::Array{Int,1}`, `size(epart) = (n_el,)`,
  host subdomain of each element.

 `npart::Array{Int,1}`, `size(npart) = (nnode,)`,
  a host subdomain of each node.

"""
function mesh_partition(cells::Array{Int,2}, ndom::Int)

  _, nel = size(cells) # Number of elements
  nnode = maximum(cells) # Number of mesh nodes

  # Write mesh for mpmetis
  open("mesh.metis", "w") do io
    print(io, "$nel\n")
    for el in 1:nel
      
      # Metis starts indexing nodes at 1 
      print(io, "$(cells[1, el]) $(cells[2, el]) $(cells[3, el])\n")
    end
  end
  
  # Call mpmetis for contiguous partition
  run(`mpmetis mesh.metis $ndom -contig`, wait=true)

  # Read and adjust partition from Metis for proper Julia indexing
  epart = reshape(readdlm("mesh.metis.epart.$ndom", Int), nel) .+ 1 
  npart = reshape(readdlm("mesh.metis.npart.$ndom", Int), nnode) .+ 1 # Metis starts indexing doms at 0

  # Clean-up
  run(`rm mesh.metis mesh.metis.epart.$ndom mesh.metis.npart.$ndom`)

  return epart, npart 
end


"""
     function save_partition(epart::Array{Int,1},
                             npart::Array{Int,1},
                             tentative_nnode::Int,
                             ndom::Int)

Saves data of mesh parition to NPZ files.

Input:

 `epart::Array{Int,1}`, `size(epart) = (n_el,)`,
  host subdomain of each element.

 `npart::Array{Int,1}`, `size(npart) = (nnode,)`,
  a host subdomain of each node.

 `tentative_nnode::Int`,
  tentative number of DoFs for the triangular mesh.

 `ndom::Int`,
  number of subdomains.

"""
function save_partition(epart::Array{Int,1},
                        npart::Array{Int,1},
                        tentative_nnode::Int,
                        ndom::Int)

  npzwrite("data/DoF$tentative_nnode-ndom$ndom.epart.npz", epart .- 1)
  npzwrite("data/DoF$tentative_nnode-ndom$ndom.npart.npz", npart .- 1)
end


"""
     function load_partition(tentative_nnode::Int, ndom::Int)

Loads data of mesh parition from NPZ files.

Input:

 `tentative_nnode::Int`,
  tentative number of DoFs for the triangular mesh.

 `ndom::Int`,
  number of subdomains.

Output:

 `epart::Array{Int,1}`, `size(epart) = (n_el,)`,
  host subdomain of each element.

 `npart::Array{Int,1}`, `size(npart) = (nnode,)`,
  a host subdomain of each node.

"""
function load_partition(tentative_nnode::Int, ndom::Int)

  epart = Array(npzread("data/DoF$tentative_nnode-ndom$ndom.epart.npz")) .+ 1
  npart = Array(npzread("data/DoF$tentative_nnode-ndom$ndom.npart.npz")) .+ 1
  return epart, npart
end


"""
     get_partition(tentative_nnode::Int,
                   ndom::Int,
                   cells::Array{Int,2},
                   cell_neighbors::Array{Int,2},
                   dirichlet_inds_g2l::Dict{Int,Int},
                   load_partition=false)

Fetches mesh partition.
"""
function get_partition(tentative_nnode::Int,
                       ndom::Int,
                       cells::Array{Int,2},
                       cell_neighbors::Array{Int,2},
                       dirichlet_inds_g2l::Dict{Int,Int},
                       load_partition=false)

  if load_existing_partition
    epart, npart = load_partition(tentative_nnode, ndom)
  else
    epart, npart = mesh_partition(cells, ndom)
    save_partition(epart, npart, tentative_nnode, ndom)
  end
  
  ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l, node_owner,
  elemd, node_Γ, node_Γ_cnt, node_Id, nnode_Id = set_subdomains(cells,
                                                                cell_neighbors,
                                                                epart, 
                                                                npart,
                                                                dirichlet_inds_g2l)

  return epart, ind_Id_g2l, ind_Γd_g2l, ind_Γ_g2l, ind_Γd_Γ2l, node_owner, node_Γ_cnt
end