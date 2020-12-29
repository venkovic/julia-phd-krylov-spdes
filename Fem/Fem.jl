module Fem

export do_isotropic_elliptic_assembly
export get_mass_matrix

export apply_dirichlet

export do_mass_covariance_assembly

export mesh_partition
export plot_TriMesh

export set_subdomains
export do_schur_assembly
export apply_schur

include("Assembly.jl")
include("BoundaryConditions.jl")
include("Mesh.jl")
include("DomainDecomposition.jl")

end