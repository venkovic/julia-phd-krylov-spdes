module Fem

# from Mesh.jl
export get_mesh,
       save_mesh,
       load_mesh
export get_total_area
export mesh_partition,
       save_partition,
       load_partition

# from BoundaryConditions.jl
export get_dirichlet_inds,
       apply_dirichlet,
       append_bc

# from EllipticPde.jl
export do_isotropic_elliptic_assembly,
       update_isotropic_elliptic_assembly!
export get_mass_matrix

# from EllipticPdeDomainDecomposition.jl,
export SubDomain
export set_subdomain
export set_subdomains

# from EllipticPdeDomainDecomposition.jl,
export prepare_global_schur,
       apply_global_schur
export prepare_local_schurs,
       assemble_local_schurs,
       apply_local_schur,
       apply_local_schurs
export assemble_A_ΓΓ_from_local_blocks

# from EllipticPdeDomainDecomposition.jl,
export get_schur_rhs,
       get_subdomain_solutions,
       merge_subdomain_solutions

# from EllipticPdeDomainDecomposition.jl,
export NeumannNeumannSchurPreconditioner,
       prepare_neumann_neumann_schur_precond,
       apply_neumann_neumann_schur

# from EllipticPdeDomainDecomposition.jl,
export LorascPreconditioner,
       prepare_lorasc_precond,
       apply_lorasc

# from EllipticPdeDomainDecomposition.jl,
export DomainDecompositionLowRankPreconditioner,
       prepare_domain_decomposition_low_rank_precond,
       apply_inv_a0, apply_inv_a0!, apply_hmat,
       apply_domain_decomposition_low_rank

# from EllipticPdeDomainDecomposition.jl,
export NeumannNeumannInducedPreconditioner,
       prepare_neumann_neumann_induced_precond,
       apply_neumann_neumann_induced

# from KarhunenLoeve.jl
export do_mass_covariance_assembly
export solve_kl

# from KarhunenLoeveDomainDecomposition.jl
export do_local_mass_covariance_assembly,
       do_local_mass_assembly
export solve_local_kl
export do_global_mass_reduced_assembly,
       do_global_mass_covariance_reduced_assembly,
       solve_global_reduced_kl
export project_on_mesh,
       trim_and_order,
       get_kl_coordinates
export draw,
       draw!

# from KarhunenLoevePllDomainDecomposition.jl, 
export pll_do_global_mass_covariance_reduced_assembly
export pll_solve_local_kl
export pll_draw

# from KarhunenLoeveDomainDecompositionHelper.jl
export suggest_parameters
export get_root_filename

# from Samplers.jl,
export McSampler,
       prepare_mc_sampler
export McmcSampler,
       prepare_mcmc_sampler


include("Mesh.jl")
include("BoundaryConditions.jl")
include("EllipticPde.jl")
include("EllipticPdeDomainDecomposition.jl")
include("KarhunenLoeve.jl")
include("KarhunenLoeveDomainDecomposition.jl")
include("KarhunenLoevePllDomainDecomposition.jl")
include("KarhunenLoeveDomainDecompositionHelper.jl")
include("Samplers.jl")
end