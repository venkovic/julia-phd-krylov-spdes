using TriangleMesh
using NPZ

function get_mesh(tentative_nnode::Int;
                  keep_segments=true)
                  
  poly = polygon_unitSquare()
  mesh = create_mesh(poly, voronoi=true, delaunay=true)
  divide_cell_into = ceil(Int, .645 * tentative_nnode)
  mesh = refine(mesh, divide_cell_into=divide_cell_into, keep_segments=keep_segments)
  return mesh
end

function save_mesh(mesh::TriangleMesh.TriMesh,
                   tentative_nnode::Int)
  npzwrite("data/DoF$tentative_nnode.cells.npz", mesh.cell' .- 1)
  npzwrite("data/DoF$tentative_nnode.points.npz", mesh.point')
  npzwrite("data/DoF$tentative_nnode.point_markers.npz", mesh.point_marker')
end

function load_mesh(tentative_nnode::Int)
  cells = Array(npzread("data/DoF$tentative_nnode.cells.npz")') .+ 1
  points = Array(npzread("data/DoF$tentative_nnode.points.npz")')
  point_markers = Array(npzread("data/DoF$tentative_nnode.point_markers.npz")')
  return cells, points, point_markers
end

function get_total_area(cells, points)
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


function mesh_partition(mesh::TriangleMesh.TriMesh, ndom::Int)
  nel = mesh.n_cell # Number of elements
  nnode = mesh.n_point # Number of mesh nodes

  # Write mesh for mpmetis
  open("mesh.metis", "w") do io
    print(io, "$nel\n")
    for el in 1:nel
      
      # Metis starts indexing nodes at 1 
      print(io, "$(mesh.cell[1, el]) $(mesh.cell[2, el]) $(mesh.cell[3, el])\n")
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
    setp(tri, linestyle = linestyle,
              linewidth = linewidth,
              marker = marker,
              markersize = markersize,
              color = color)
    
    fig[:canvas][:draw]()
    println("yrdy")    
    return fig
end
"""