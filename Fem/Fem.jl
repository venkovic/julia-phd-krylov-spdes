module Fem

export do_isotropic_elliptic_assembly
export do_mass_covariance_assembly
export get_mass_matrix
export apply_dirichlet
export mesh_partition
export set_subdomains

export plot_TriMesh

include("Assembly.jl")
include("BoundaryConditions.jl")
include("Mesh.jl")
include("DomainDecomposition.jl")

end