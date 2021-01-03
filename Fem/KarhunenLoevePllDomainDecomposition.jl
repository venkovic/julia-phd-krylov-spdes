using Distributed
import Arpack

struct SubDomain
    inds_g2l::Dict{Int,Int}  #
    inds_l2g::Vector{Int}    #
    elems::Vector{Int}       #
    ϕ::Array{Float64,2}      #
end

function pll_do_global_mass_covariance_reduced_assembly(cells::Array{Int,2},
                                                        points::Array{Float64,2},
                                                        domain::Dict{Int,SubDomain},
                                                        idom::Int,
                                                        md::Array{Int,1},
                                                        cov)

    _, nel = size(cells) # Number of elements
    _, nnode = size(points) # Number of nodes
    ndom = length(domain) # Number of subdomains 
    x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
    Δx, Δy = zeros(3), zeros(3), zeros(3) # Used to store terms of shoelace formula
    
    # Global reduced mass covariance matrix
    md_sum = sum(md)
    K = zeros(md_sum, md_sum)
    
    # Loop over other subdomains
    nnode_idom = domain[idom].inds_g2l.count
    for jdom in 1:ndom
      nnode_jdom = domain[jdom].inds_g2l.count

      R = zeros(nnode_idom, nnode_jdom) # R[i, j] ≈ ∑_e ∫_{Ω'_e} ϕ_i(P') cov(P', P_j) dΩ'
      C = zeros(nnode_idom, nnode_jdom) # C[i, j] ≈ ∫_Ω ϕ_i(P) ∫_{Ω'} cov(P, P') ϕ_j(P') dΩ' dΩ.
      
      # Loop over mesh nodes of the jdom-th subdomain 
      for (j, jnode) in enumerate(domain[jdom].inds_l2g)
        
        # Get coordinates of node
        xj = points[1, jnode]
        yj = points[2, jnode]
        
        # Loop over elements of the idom-th subdomain
        for (iel, el) in enumerate(domain[idom].elems)
          
          # Get (x, y) coordinates of each element vertex
          for r in 1:3
            rnode = cells[r, el]
            x[r], y[r] = points[1, rnode], points[2, rnode]
          end
          
          # Terms of the shoelace formula for a triangle
          Δx[1] = x[3] - x[2]
          Δx[2] = x[1] - x[3]
          Δx[3] = x[2] - x[1]
          Δy[1] = y[2] - y[3]
          Δy[2] = y[3] - y[1]
          Δy[3] = y[1] - y[2]
          
          # Area of element
          Area_el = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
          
          # Add local contributions
          for r in 1:3
            s = r + 1 - floor(Int, (r + 1) / 3) * 3
            s == 0 ? s = 3 : nothing
            t = r + 2 - floor(Int, (r + 2) / 3) * 3
            t == 0 ? t = 3 : nothing
            i = domain[idom].inds_g2l[cells[r, el]]
            R[i, j] += (2 * cov(x[r], y[r], xj, yj) 
                          + cov(x[s], y[s], xj, yj) 
                          + cov(x[t], y[t], xj, yj)) * Area_el / 12
          end
        end # for el
      end # for jnode
        
        
      # Loop over elements of the jdom-th subdomain
      for (jel, el) in enumerate(domain[jdom].elems)
          
        # Get (x, y) coordinates of each element vertex
        for r in 1:3
          rnode = cells[r, el]
          x[r], y[r] = points[1, rnode], points[2, rnode]
        end
          
        # Terms of the shoelace formula for a triangle
        Δx[1] = x[3] - x[2]
        Δx[2] = x[1] - x[3]
        Δx[3] = x[2] - x[1]
        Δy[1] = y[2] - y[3]
        Δy[2] = y[3] - y[1]
        Δy[3] = y[1] - y[2]
          
        # Area of element
        Area_el = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
          
        # Add local contributions
        for r in 1:3
          s = r + 1 - floor(Int, (r + 1) / 3) * 3
          s == 0 ? s = 3 : nothing
          t = r + 2 - floor(Int, (r + 2) / 3) * 3
          t == 0 ? t = 3 : nothing
          j = domain[jdom].inds_g2l[cells[r, el]]
          k = domain[jdom].inds_g2l[cells[s, el]]
          ℓ = domain[jdom].inds_g2l[cells[t, el]]


          # Loop over mesh nodes of the idom-th subdomain 
          for (i, inode) in enumerate(domain[idom].inds_l2g)
            C[i, j] += (2 * R[i, j] 
                          + R[i, k]  
                          + R[i, ℓ]) * Area_el / 12
          end
        end # for r
      end # for jel
      
      # Loop over local modes of the idom-th subdomain
      for α in 1:md[idom]
        if idom == 1
          ind_α_idom = α
        else
          ind_α_idom = sum(md[1:idom-1]) + α 
        end

        # Loop over local modes of the jdom-th subdomain
        for β in 1:md[jdom]
          if jdom == 1
            ind_β_jdom = β
          else
            ind_β_jdom = sum(md[1:jdom-1]) + β
          end

          # Add contributions
          for j in 1:nnode_jdom
            Φ_j_β = domain[jdom].ϕ[j, β]

            for i in 1:nnode_idom
              K[ind_α_idom, ind_β_jdom] += domain[idom].ϕ[i, α] * C[i, j] * Φ_j_β
            end
          end
        end # end β
      end # end α
    end # for jdom
  
    println("Done with idom = $idom.")
  
    return K
  end


function pll_solve_local_kl(mesh::TriangleMesh.TriMesh,
                            epart::Array{Int,1},
                            cov::Function,
                            nev::Int,
                            idom::Int)

  ndom = maximum(epart) # Number of subdomains

  # Parallel loop over subdomains
  inds_l2g, inds_g2l, elems = set_subdomain(mesh, epart, idom)
  
  # Assemble local generalized eigenvalue problem
  C = do_local_mass_covariance_assembly(mesh.cell, mesh.point, inds_l2g, 
                                        inds_g2l, elems, cov)
  M = do_local_mass_assembly(mesh.cell, mesh.point, inds_g2l, elems)
  
  # Solve local generalized eigenvalue problem
  λ, ϕ = map(x -> real(x), Arpack.eigs(C, M, nev=nev))
  
  # Arpack 0.5.1 does not normalize the vectors properly
  for k in 1:size(ϕ)[2]
    ϕ[:, k] ./= sqrt(ϕ[:, k]'M * ϕ[:, k])
  end
  
  # truncate here
  println("$idom, $(length(inds_l2g)), $(sum(λ))")
  return Dict(idom => SubDomain(inds_g2l, inds_l2g, elems, ϕ))
end


function project_on_mesh(mesh::TriangleMesh.TriMesh,
                         Φ::Array{Float64,2},
                         domain:: Dict{Int,SubDomain})

  ndom = length(domain) # Number of subdomains
  _, nmodes = size(Φ) # Number of reduced modes
  md = Int[] # Number of local modes in each subdomain
  nnodes = mesh.n_point # Number of mesh nodes
  Ψ = zeros(nnodes, nmodes) # Eigenfunction projection at the mesh nodes
  cnt = zeros(Int, nnodes) # Number of subdomains to which each node belongs

  # Get the number of local modes retained for each subdomain
  for idom in 1:ndom
    push!(md, size(domain[idom].ϕ)[2])
  end
  
  # Loop over reduced modes
  for imode in 1:nmodes
    # Loop over subdomains
    cnt .= 0
    for idom in 1:ndom

      # Loop over mesh nodes
      for (i, inode) in enumerate(domain[idom].inds_l2g)
        cnt[inode] += 1

        # Loop over local modes and add contributions
        for α in 1:md[idom]
          idom == 1 ? ind_α_idom = α : ind_α_idom = sum(md[1:idom-1]) + α
          Ψ[inode, imode] += Φ[ind_α_idom, imode] * domain[idom].ϕ[i, α]
        end 
      end # for inode
    end # for idom

    Ψ[:, imode] ./= cnt
  end # for imode

  return Ψ
end
















function pll_draw(mesh, Λ, Φ, ϕd, inds_l2gd)
    nnode = mesh.n_point # Number of mesh nodes
    nmode = length(Λ) # Number of global modes
    ndom = length(ϕd) # Number of subdomains
    md = Int[] # Number of local modes for each subdomain
    
    ξ = rand(Distributions.Normal(), nmode)
    g = zeros(nnode)
    cnt = zeros(Int, nnode)
    
    # Get the number of local modes retained for each subdomain
    for idom in 1:ndom
      push!(md, size(ϕd[idom])[2])
    end
    
    for idom in 1:ndom
      for (i, inode) in enumerate(inds_l2gd[idom])
        cnt[inode] += 1
        for α in 1:md[idom]
          idom == 1 ? ind_α_idom = α : ind_α_idom = sum(md[1:idom-1]) + α
          for γ in 1:nmode
            g[inode] += sqrt(Λ[γ]) * ξ[γ] * Φ[ind_α_idom, γ] * ϕd[idom][i, α]
          end
        end # for α
      end # for γ
    end # for idom
    g ./= cnt # should done cleanly instead
    return ξ, g
end