struct SubDomain
  inds_g2l::Dict{Int,Int}  #
  inds_l2g::Vector{Int}    #
  elems::Vector{Int}       #
  ϕ::Array{Float64,2}      #
  center::Array{Float64,1} #
  energy::Float64          #
end

function pll_do_global_mass_covariance_reduced_assembly(cells::Array{Int,2},
                                                        points::Array{Float64,2},
                                                        domain::Union{Dict{Int,SubDomain},
                                                                      Array{SubDomain,1}},
                                                        idom::Int,
                                                        md::Array{Int,1},
                                                        cov::Function;
                                                        forget=-1.)

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
    for jdom in idom:ndom
      
      # Check if subdomains are significantly correlated
      if cov(domain[idom].center[1], domain[idom].center[2], 
             domain[jdom].center[1], domain[jdom].center[2]) > forget
        
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
    end # if subdomain are significantly correlated 
  end # for jdom
  
  println("Done with idom = $idom.")
  
  return K
end


function pll_solve_local_kl(cells::Array{Int,2},
                            points::Array{Float64,2},
                            epart::Array{Int,1},
                            cov::Function,
                            nev::Int,
                            idom::Int;
                            relative=.99,
                            pll=:pmap)

  ndom = maximum(epart) # Number of subdomains

  # Parallel loop over subdomains
  inds_l2g, inds_g2l, elems, center = set_subdomain(cells, points, epart, idom)
  
  # Assemble local generalized eigenvalue problem
  C = do_local_mass_covariance_assembly(cells, points, inds_l2g, 
                                        inds_g2l, elems, cov)
  M = do_local_mass_assembly(cells, points, inds_g2l, elems)
  
  # Solve local generalized eigenvalue problem
  λ, ϕ = map(x -> real(x), Arpack.eigs(C, M, nev=nev))
  #λ, ϕ  = LinearAlgebra.eigen(C, Array(M), sortby=λ->-λ)

  # Arpack 0.5.1 does not normalize the vectors properly
  for k in 1:size(ϕ)[2]
    ϕ[:, k] ./= sqrt(ϕ[:, k]'M * ϕ[:, k])
  end
  
  # Integrate variance over subdomain
  x, y = zeros(3), zeros(3)
  Δx, Δy = zeros(3), zeros(3)
  Area = 0.
  for el in elems
    for r in 1:3
      rnode = cells[r, el]
      x[r], y[r] = points[1, rnode], points[2, rnode]
    end  
    Δx[1] = x[3] - x[2]
    Δx[2] = x[1] - x[3]
    Δx[3] = x[2] - x[1]
    Δy[1] = y[2] - y[3]
    Δy[2] = y[3] - y[1]
    Δy[3] = y[1] - y[2]   
    Area += (Δx[3] * Δy[2] - Δx[2] * Δy[3]) / 2.
  end
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

  # Details about truncation
  str = "idom = $idom, $nvec/$nev vectors kept for "
  str *= @sprintf "%.5f" (energy_achieved / energy_expected * relative)
  println("$str relative energy")

  subdomain = SubDomain(inds_g2l,
              inds_l2g,
              elems,
              ϕ[:, 1:nvec],
              center,
              energy_expected / relative)

  if pll == :static_scheduling
    return Dict(idom => subdomain)
  else 
    return subdomain
  end
end


function solve_global_reduced_kl(nnode::Int,
                                 K::Array{Float64,2},
                                 energy_expected::Float64,
                                 domain::Union{Dict{Int,SubDomain},
                                               Array{SubDomain,1}};
                                 relative=.99)
    
  Ksym = LinearAlgebra.Symmetric(K)
  Λ, Φ = LinearAlgebra.eigen(Ksym)
  Λ, Φ = trim_and_order(Λ, Φ)
  energy_expected *= relative
  energy_achieved = 0.
  nvec = 0
  for λ_i in Λ
    energy_achieved += λ_i
    nvec += 1 
    energy_achieved >= energy_expected ? break : nothing
  end
  Ψ = project_on_mesh(nnode, Φ[:, 1:nvec], domain)

  # Details about truncation
  str = "$nvec/$(length(Λ)) vectors kept for "
  str *= @sprintf "%.5f" (energy_achieved / energy_expected * relative)
  println("$str relative energy")

  return Λ[1:nvec], Ψ
end


"""
     pll_compute_kl(ndom::Int,
                    nev::Int,
                    tentative_nnode::Int,
                    cov::Function,
                    root_fname::String;
                    forget=1e-6,
                    load_existing_mesh=false,
                    load_existing_partition=false,
                    prebatch=true,
                    verbose=true)
  
Function called by the master node to compute a Karhunen-Loeve (KL) by distributed
domain decomposition.
  
Input:

 `ndom::Int`, `ndom > 1`,
  number of subdomains.

 `nev::Int`, 
  maximum number of eigenpairs computed for each local KL expansion.

 `tentative_nnode`,
  approximate number of DoFs wanted.

 `cov::Function`,
  covariance function, must be available everywhere.

 `rootfname::String`
  filename's root.

 `forget=1e-6`,
  threshold of covariance between points in distinct subdomains under which
  subdomain pairs are ignored for the assembly of the reduced global mass
  covariance matrix. Note that `forget<0` ⟹ all pairs are considered.

 `load_existing_mesh=false`.

 `load_existing_partition=false`.


 `pll=:dynamic_scheduling`, `pll ∈ (:dynamic_scheduling, :pmap, :static_scheduling)`,
  refers to how the work load of the distributed loops is handled.
  If `prebatch==true`, @distributed (op) for loops are used, i.e., the load of each 
  worker is planned ahead of the loop's execution, which works well for cases with 
  a small `tentative_nnode`. For cases with a larger `tentative_nnode`, unexpected 
  worker termination is more likely to happen, in which case the whole computation 
  fails. We recommend using `prebatch=false`, which consists of using pmap, is more 
  fault tolerant (? and dynamically dispatches loads to workers ?). There is no julia
  implementation of distributed mapreduce. Therefore, We need to store local contri-
  -butions to the global reduced mass covariance matrix. Fortunately, the number of
  reduced kl modes does not increase with tentative_nnode.
  
  `verbose=true`.

Output:

 `ind_Id_g2l::Array{Dict{Int,Int}}`, 
  conversion tables from global to local indices of nodes strictly inside each subdomain.

 `ind_Γd_g2l::Array{Dict{Int,Int}}`,
  conversion table from global to local indices of nodes on the interface of each subdomain.

"""
function pll_compute_kl(ndom::Int,
                        nev::Int,
                        tentative_nnode::Int,
                        cov::Function,
                        root_fname::String;
                        forget=1e-6,
                        load_existing_mesh=false,
                        load_existing_partition=false,
                        pll=:pmap,
                        verbose=true)
   
  if load_existing_mesh
    cells, points, point_markers, cell_neighbors = load_mesh(tentative_nnode)
    _, nnode = size(points)
  else
    mesh = get_mesh(tentative_nnode)
    save_mesh(mesh, tentative_nnode)
    cells = mesh.cell
    points = mesh.point
    point_markers = mesh.point_marker
    cell_neighbors = mesh.cell_neighbor
    _, nnode = size(points)
  end

  if load_existing_partition
    epart, npart = load_partition(tentative_nnode, ndom)
  else
    epart, npart = mesh_partition(cells, ndom)
    save_partition(epart, npart, tentative_nnode, ndom)
  end

  # Broadcast
  @everywhere begin
    ndom = $ndom
    nev = $nev
    tentative_nnode = $tentative_nnode
    forget = $forget
    nnode = $nnode
    cells = $cells
    points = $points
    epart = $epart
  end  

  if verbose
    space_println("nnode = $(size(points)[2])")
    space_println("nel = $(size(cells)[2])")
  end

  # Get cutoff energy levels
  relative_local, relative_global = suggest_parameters(nnode)

  verbose ? printlnln("pll_solve_local_kl ...") : nothing
  if pll == :static_scheduling
    @time domain = @sync @distributed merge! for idom in 1:ndom
      pll_solve_local_kl(cells, points, epart, cov, nev, idom, 
                         relative=relative_local,
                         pll=pll)
    end

  elseif pll in (:pmap, :dynamic_scheduling)
    domain = pmap(idom -> pll_solve_local_kl(cells,
                                             points,
                                             epart,
                                             cov,
                                             nev,
                                             idom,
                                             relative=relative_local,
                                             pll=pll),
                  1:ndom)
  end
  verbose ? println("... done with pll_solve_local_kl.") : nothing

  energy_expected = 0.
  for idom in 1:ndom
    energy_expected += domain[idom].energy
  end

  # Store number of local modes retained for each subdomain
  md = zeros(Int, ndom) 
  for idom in 1:ndom
    md[idom] = size(domain[idom].ϕ)[2]
  end

  # Broadcast
  @everywhere md = $md

  verbose ? printlnln("pll_do_global_mass_covariance_reduced_assembly ...") : nothing
  @time begin
    
    if pll == :static_scheduling
      K = @sync @distributed (+) for idom in 1:ndom
      pll_do_global_mass_covariance_reduced_assembly(cells, points, 
                                                     domain, idom, md, cov,
                                                     forget=forget)
      end

    elseif pll == :pmap
      Kd = pmap(idom -> pll_do_global_mass_covariance_reduced_assembly(cells,
                                                                       points,
                                                                       domain,
                                                                       idom,
                                                                       md, 
                                                                       cov,
                                                                       forget=forget),
                1:ndom)
      K = reduce(+, Kd)

    elseif pll == :dynamic_scheduling
      nothing

    end # if

  end # @time
  verbose ? println("... done with pll_do_global_mass_covariance_reduced_assembly.") : nothing
  
  verbose ? printlnln("solve_global_reduced_kl ...") : nothing
  Λ, Ψ = @time solve_global_reduced_kl(nnode, K, energy_expected, domain, 
                                       relative=relative_global)
  verbose ? println("... done with do_global_mass_covariance_reduced_assembly.") : nothing

  npzwrite("data/$root_fname.kl-eigvals.npz", Λ)
  npzwrite("data/$root_fname.kl-eigvecs.npz", Ψ)

  return Λ, Ψ
end


function project_on_mesh(nnode::Int,
                         Φ::Array{Float64,2},
                         domain::Union{Dict{Int,SubDomain},
                                       Array{SubDomain,1}})

  ndom = length(domain) # Number of subdomains
  _, nmodes = size(Φ) # Number of reduced modes
  md = Int[] # Number of local modes in each subdomain
  Ψ = zeros(nnode, nmodes) # Eigenfunction projection at the mesh nodes
  cnt = zeros(Int, nnode) # Number of subdomains to which each node belongs

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


function pll_draw(nnode::Int, 
                  Λ::Array{Float64,1},
                  Φ::Array{Float64,2}, 
                  ϕd::Array{Array{Float64,2},1}, 
                  inds_l2gd::Array{Array{Int,1},1})

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