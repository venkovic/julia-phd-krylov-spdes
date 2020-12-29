using TriangleMesh
using PyPlot


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