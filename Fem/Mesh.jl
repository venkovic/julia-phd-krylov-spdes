using TriangleMesh

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
  nel = mesh.n_cell # number of elements
  
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
  epart = readdlm("mesh.metis.epart.$ndom", Int) .+ 1 # Metis starts indexing doms at 0
  npart = readdlm("mesh.metis.npart.$ndom", Int) .+ 1 # Metis starts indexing doms at 0
  
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