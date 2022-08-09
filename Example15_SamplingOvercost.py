import numpy as np
import pylab as pl
from postproc_utils import to_axis, get_root_filename

nKL = 500
n_mcmc = (50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
u_values = {m: [] for m in n_mcmc}
repeats = {m: [] for m in n_mcmc}
u_mcmc_chains = {m: [] for m in n_mcmc}
cov_u = {m: [] for m in n_mcmc}
t_decor = {m: [] for m in n_mcmc}
gamma_u = {m: [] for m in n_mcmc}

nchains = 10
nsmp = 10_000

tentative_nnode = 4_000
model = 'SExp'
L = .1
sig2 = 1.
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

u_sols_mc = np.load("data/%s.u_values_mc.npz" % root_fname)

for m in n_mcmc:
  for ichain in range(nchains):
    if m >= nKL:
      u_values_of_m = np.load("data/%s.u_values_mcmc.chain%d.npz" % (root_fname, ichain))
      repeats_of_m = np.load("data/%s.repeats_mcmc.chain%d.npz" % (root_fname, ichain))
    else:
      u_values_of_m = np.load("data/%s.u_values_hybrid%d.chain%d.npz" % (root_fname, m, ichain))
      repeats_of_m = np.load("data/%s.repeats_hybrid%d.chain%d.npz" % (root_fname, m, ichain))
    u_values[m] += [u_values_of_m]
    repeats[m] += [repeats_of_m]
    vals = []
    for i, u in enumerate(u_values_of_m):
      vals += repeats_of_m[i] * [u]
    u_mcmc_chains[m] += [np.array(vals)]
    #
    fft_u = np.fft.fft(u_mcmc_chains[m][-1]-u_mcmc_chains[m][-1].mean())
    nvals = len(fft_u)
    cov_u[m] += [np.real(np.fft.ifft(fft_u*np.conjugate(fft_u))/nvals)[:nvals // 2]]
    t_decor[m] += [np.where(cov_u[m][-1] < 0)[0][0]]
    #
    p_acc = nsmp / np.sum(repeats_of_m)
    gamma_u[m] += [p_acc * (np.var(u_sols_mc) + 2. * np.sum(cov_u[m][-1][1:t_decor[m][-1]])) / np.var(u_sols_mc)]


for m in n_mcmc:
  print(m, " ", np.mean(gamma_u[m]), np.std(gamma_u[m]))