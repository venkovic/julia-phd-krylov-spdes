module Fem

export SubDomain

export do_isotropic_elliptic_assembly
export get_mass_matrix

export get_dirichlet_inds
export apply_dirichlet
export append_bc

export do_mass_covariance_assembly
export solve_kl

export get_mesh
export save_mesh
export load_mesh
export mesh_partition
export save_partition
export load_partition
export get_total_area

export set_subdomain
export set_subdomains

export do_schur_assembly
export apply_schur
export get_schur_rhs
export get_subdomain_solutions
export merge_subdomain_solutions

export do_local_mass_covariance_assembly
export do_local_mass_assembly
export do_global_mass_reduced_assembly
export do_global_mass_covariance_reduced_assembly
export solve_local_kl
export solve_global_reduced_kl
export project_on_mesh
export draw
export draw!
export trim_and_order
export get_kl_coordinates

export pll_do_global_mass_covariance_reduced_assembly
export pll_solve_local_kl
export pll_draw

export suggest_parameters
export get_root_filename

include("Mesh.jl")
include("EllipticPde.jl")
include("BoundaryConditions.jl")
include("EllipticPdeDomainDecomposition.jl")
include("KarhunenLoeve.jl")
include("KarhunenLoeveDomainDecomposition.jl")
include("KarhunenLoevePllDomainDecomposition.jl")
include("KarhunenLoeveDomainDecompositionHelper.jl")
end