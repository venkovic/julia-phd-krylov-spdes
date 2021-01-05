module Fem

export SubDomain

export do_isotropic_elliptic_assembly
export get_mass_matrix

export apply_dirichlet

export do_mass_covariance_assembly

export get_mesh
export mesh_partition
export get_total_area

export set_subdomain
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