import numpy as np
import pylab as pl
from postproc_utils import to_axis, get_root_filename, get_root_filename, plot_spectra, plot_iters_vs_dofs, plot_iters_vs_ndoms

tentative_nnode = 4_000 # 4_000, 16_000, 32_000, 64_000, 128_000
model = 'SExp'
L = .1 # .1, .5
sig2 = 1. # .5, 1.
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

nreals = 3
ndoms = [5, 10, 20, 30, 80, 200]#[5, 10, 20, 30, 80, 200]

ndom = 200
tentative_nnodes = (4_000, 8_000, 16_000, 32_000, 64_000, 128_000,)
root_fnames = {nnode: get_root_filename(model, sig2, L, nnode) for nnode in tentative_nnodes}

preconds = ('neumann-neumann_0', 'neumann-neumann_t', 'A_GG_0', 'A_GG_t',)
eigvals  = {precond: {indom: [] for indom in ndoms} for precond in preconds}
for indom in ndoms:
  tags = ('neumann-neumann_ndom%d_0' % indom,
          'neumann-neumann_ndom%d_t' % indom,
          'A_GG_ndom%d_0' % indom,
          'A_GG_ndom%d_t' % indom,)
  for i, precond in enumerate(preconds):
    tag = tags[i]
    for ireal in range(nreals):
      eigvals_ld = np.load("data/%s.%s_Ss%d.ld.eigvals.npz" % (root_fname, tag, ireal+1))
      eigvals_md = np.load("data/%s.%s_Ss%d.md.eigvals.npz" % (root_fname, tag, ireal+1))
      eigvals[precond][indom] += [np.concatenate((eigvals_ld, eigvals_md))]

plot_spectra(eigvals, nreals, root_fname, pb='Example07')


iters = {precond: {nnode: 0 for nnode in tentative_nnodes} for precond in preconds}
tags = ('neumann-neumann_ndom%d_0' % ndom,
        'neumann-neumann_ndom%d_t' % ndom,
        'A_GG_ndom%d_0' % ndom,
        'A_GG_ndom%d_t' % ndom,)
for nnode in tentative_nnodes:
  fname = root_fnames[nnode]
  for i, precond in enumerate(preconds):
    tag = tags[i]
    iters[precond][nnode] = np.mean(np.load('data/%s.%s.pcg-iters.nreals1000.npz' % (fname, tag)))

plot_iters_vs_dofs(iters, pb='Example07')


iters2 = {precond: {indom: 0 for indom in ndoms} for precond in preconds}
for indom in ndoms:
  fname = root_fname
  tags = ('neumann-neumann_ndom%d_0' % indom,
          'neumann-neumann_ndom%d_t' % indom,
          'A_GG_ndom%d_0' % indom,
          'A_GG_ndom%d_t' % indom,)
  for i, precond in enumerate(preconds):
    tag = tags[i]
    iters2[precond][indom] = np.mean(np.load('data/%s.%s.pcg-iters.nreals1000.npz' % (fname, tag)))

plot_iters_vs_ndoms(iters2, pb='Example07')