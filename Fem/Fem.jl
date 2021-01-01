module Fem

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

export set_subdomains
export do_schur_assembly
export apply_schur

include("Mesh.jl")
include("EllipticPde.jl")
include("BoundaryConditions.jl")
include("EllipticPdeDomainDecomposition.jl")
include("KarhunenLoeve.jl")
include("KarhunenLoeveDomainDecomposition.jl")

end