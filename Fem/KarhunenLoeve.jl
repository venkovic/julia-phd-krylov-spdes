"""
     do_mass_covariance_assembly(cells::Array{Int,2},
                                 points::Array{Float64,2},
                                 cov::Function)

Assembles Galerkin kernel operator generated by the covariance function
cov with 2D P1 finite elements with a given triangulation (cells, points).

Input:

 `cells::Array{Int,2}`, `size(cells) = (3, n_el)`,
  nodes of each element.

 `points::Array{Float64,2}`, `size(cells) = (2, nnode)`,
  (x,y)-coordinates of all mesh nodes.

 `cov::Function`,
  covariance function.

Output:

 `C::Array{Float64,2}`,
  array of Galerkin formulation (nnode-by-nnode)
  with components C_ij = ∫_Ω ϕ_i(P) ∫_{Ω'} cov(P, P') ϕ_j(P') dΩ' dΩ.

"""
function do_mass_covariance_assembly(cells::Array{Int,2},
                                     points::Array{Float64,2},
                                     cov::Function)

  _, nel = size(cells) # Number of elements
  _, nnode = size(points) # Number of nodes
  R = zeros(nnode, nnode) # R[i, j] ≈ ∑_e ∫_{Ω'_e} ϕ_i(P') cov(P', P_j) dΩ'
  C = zeros(nnode, nnode) # C[i, j] ≈ ∫_Ω ϕ_i(P) ∫_{Ω'} cov(P, P') ϕ_j(P') dΩ' dΩ.
  x, y = zeros(3), zeros(3) # (x, y) coordinates of element vertices
  Δx, Δy = zeros(3), zeros(3), zeros(3) # Used to store terms of shoelace formula
  Area = zeros(nel) # Used to store element areas
  
  # Loop over mesh nodes
  for j in 1:nnode
    
    # Get coordinates of node
    xj = points[1, j]
    yj = points[2, j]
    
    # Loop over elements
    for iel in 1:nel
      
      # Get (x, y) coordinates of each element vertex
      # and coefficient at the center of the element
      for r in 1:3
        rr = cells[r, iel]
        x[r], y[r] = points[1, rr], points[2, rr]
      end
      
      # Terms of the shoelace formula for a triangle
      Δx[1] = x[3] - x[2]
      Δx[2] = x[1] - x[3]
      Δx[3] = x[2] - x[1]
      Δy[1] = y[2] - y[3]
      Δy[2] = y[3] - y[1]
      Δy[3] = y[1] - y[2]
      
      # Area of element
      Area_iel = (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
      Area[iel] = Area_iel
      
      # Add local contributions
      for r in 1:3
        s = r + 1 - floor(Int, (r + 1) / 3) * 3
        s == 0 ? s = 3 : nothing
        t = r + 2 - floor(Int, (r + 2) / 3) * 3
        t == 0 ? t = 3 : nothing
        i = cells[r, iel]
        R[i, j] += (2 * cov(x[r], y[r], xj, yj) 
                      + cov(x[s], y[s], xj, yj) 
                      + cov(x[t], y[t], xj, yj)) * Area_iel / 12
      end
    end
  end
  
  # Loop over mesh nodes
  for i in 1:nnode
    
    # Loop over elements
    for iel in 1:nel
      
      # Get area of element
      Area_iel = Area[iel]
      
      # Add local contributions
      for r in 1:3
        s = r + 1 - floor(Int, (r + 1) / 3) * 3
        s == 0 ? s = 3 : nothing
        t = r + 2 - floor(Int, (r + 2) / 3) * 3
        t == 0 ? t = 3 : nothing
        j = cells[r, iel]
        k = cells[s, iel]
        ℓ = cells[t, iel]
        C[i, j] += (2 * R[i, j] 
                      + R[i, k]  
                      + R[i, ℓ]) * Area_iel / 12
      end
    end
  end
  
  return C
end


"""
     solve_kl(cells::Array{Int,2},
              points::Array{Float64,2},
              cov::Function,
              nev::Int;
              relative=.99,
              verbose=false)

Solves generalized eigenvalue problems of local 
Karhunen Loeve expansions.

"""
function solve_kl(cells::Array{Int,2},
                  points::Array{Float64,2},
                  cov::Function,
                  nev::Int;
                  relative=.99,
                  verbose=false)
  
  _, n_el = size(cells) # number of elements

  if verbose
    print("assemble covariance mass matrix ...")
    C = @time do_mass_covariance_assembly(cells, points, cov)
    print("assemble rhs mass matrix ...")
    M = @time get_mass_matrix(cells, points)
    print("solve for $nev dominant eigenpairs ...")
    λ, ϕ = map(x -> real(x), Arpack.eigs(C, M, nev=nev))
  else    
    C = @time do_mass_covariance_assembly(cells, points, cov)
    M = @time get_mass_matrix(cells, points)
    λ, ϕ = map(x -> real(x), Arpack.eigs(C, M, nev=nev))
  end

  # Integrate variance over domain
  center = zeros(2)
  x, y = zeros(3), zeros(3)
  Δx, Δy = zeros(3), zeros(3)
  Area = 0.
  for el in 1:n_el
    for r in 1:3
      rnode = cells[r, el]
      x[r], y[r] = points[1, rnode], points[2, rnode]
      center[1] += points[1, rnode] / 3
      center[2] += points[2, rnode] / 3
    end  
    Δx[1] = x[3] - x[2]
    Δx[2] = x[1] - x[3]
    Δx[3] = x[2] - x[1]
    Δy[1] = y[2] - y[3]
    Δy[2] = y[3] - y[1]
    Δy[3] = y[1] - y[2]   
    Area += (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
  end
  center ./= n_el
  energy_expected = relative * Area * cov(center[1], center[2],
                    center[1], center[2])

  # Keep dominant eigenpairs with positive eigenvalues
  energy_achieved = 0.
  nvec = 0
  for i in 1:nev
    if λ[i] > 0 
      nvec += 1
      energy_achieved += λ[i]
    else
      break
    end
    energy_achieved >= energy_expected ? break : nothing
  end

  # Arpack 0.5.1 does not normalize the vectors properly
  for k in 1:nvec
    ϕ[:, k] ./= sqrt(ϕ[:, k]'M * ϕ[:, k])
  end

  # Details about truncation
  str = "$nvec/$nev vectors kept for "
  str *= @sprintf "%.5f" (energy_achieved / energy_expected * relative)
  println("$str relative energy")

  return λ[1:nvec], ϕ[:, 1:nvec]
end