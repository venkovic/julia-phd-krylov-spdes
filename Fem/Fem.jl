module Fem

export SubDomain

export do_isotropic_elliptic_assembly
export get_mass_matrix

export apply_dirichlet

export do_mass_covariance_assembly

export mesh_partition
export get_total_area

export set_subdomain
export do_local_mass_covariance_assembly
export do_local_mass_assembly
export do_global_mass_reduced_assembly
export do_global_mass_covariance_reduced_assembly
export draw

export do_global_mass_covariance_reduced_pll_assembly
export pll_draw

include("Mesh.jl")
include("EllipticPde.jl")
include("BoundaryConditions.jl")
include("EllipticPdeDomainDecomposition.jl")
include("KarhunenLoeve.jl")
include("KarhunenLoeveDomainDecomposition.jl")
include("KarhunenLoevePllDomainDecomposition.jl")
end