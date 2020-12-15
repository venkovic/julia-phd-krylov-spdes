using TriangleMesh
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