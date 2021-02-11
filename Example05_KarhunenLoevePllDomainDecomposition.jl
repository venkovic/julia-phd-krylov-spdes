import Pkg
using Distributed
Pkg.activate(".")

push!(LOAD_PATH, "./Utils/")

using Utils: space_println, printlnln,
             add_my_procs,
             dynamic_mapreduce!

machines = ["",]
add_my_procs(machines, Sys.CPU_THREADS)

@everywhere begin
  import Pkg
  Pkg.activate(".")
end

@everywhere begin
  push!(LOAD_PATH, "./Utils/")
  push!(LOAD_PATH, "./Fem/")
end

@everywhere begin 
  using Fem
  using Distributed
end

using NPZ: npzwrite

ndom = 40
nev = 40
tentative_nnode = 20_000
forget = 1e-6

model = "SExp"
sig2 = 1.
L = .1


@everywhere cov = (x1, y1, x2, y2) -> cov_sexp(x1, y1, x2, y2, sig2, L)

root_fname = get_root_filename(model, sig2, L, tentative_nnode)

Λ, Ψ = pll_compute_kl(ndom,
                      nev,
                      tentative_nnode,
                      cov,
                      root_fname,
                      pll=:dynamic_scheduling)

printlnln("sample ...")
ξ, g = @time draw(Λ, Ψ)

printlnln("sample in place ...")
@time draw!(Λ, Ψ, ξ, g)
npzwrite("data/$root_fname.greal.npz", g)