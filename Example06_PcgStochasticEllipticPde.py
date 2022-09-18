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

preconds = ('amg_0', 'amg_t', 'bJ_0', 'bJ_t', 'lorasc_eps00_0', 'lorasc_eps00_t', 'lorasc_eps01_0', 'lorasc_eps01_t',)
eigvals  = {precond: {indom: [] for indom in ndoms} for precond in preconds}
for indom in ndoms:
  tags = ('amg_0',
          'amg_t',
          'bJ_nb%d_0' % indom,
          'bJ_nb%d_t' % indom,
          'lorasc_ndom%d_eps00_0' % indom,
          'lorasc_ndom%d_eps00_t' % indom,
          'lorasc_ndom%d_eps01_0' % indom,
          'lorasc_ndom%d_eps01_t' % indom,)
  for i, precond in enumerate(preconds):
    tag = tags[i]
    for ireal in range(nreals):
      eigvals_ld = np.load("data/%s.%s_As%d.ld.eigvals.npz" % (root_fname, tag, ireal+1))
      eigvals_md = np.load("data/%s.%s_As%d.md.eigvals.npz" % (root_fname, tag, ireal+1))
      eigvals[precond][indom] += [np.concatenate((eigvals_ld, eigvals_md))]

plot_spectra(eigvals, nreals, root_fname, pb='Example06')


preconds = ('amg_0', 'amg_t', 'bJ_0', 'bJ_t', 'lorasc_eps00_0', 'lorasc_eps00_t', 'lorasc_eps01_0', 'lorasc_eps01_t',)
iters = {precond: {nnode: 0 for nnode in tentative_nnodes} for precond in preconds}
tags = ('amg_0', 'amg_t',
        'bJ_nb%d_0' % ndom,
        'bJ_nb%d_t' % ndom,
        'lorasc_ndom%d_eps00_0' % ndom,
        'lorasc_ndom%d_eps00_t' % ndom,
        'lorasc_ndom%d_eps01_0' % ndom,
        'lorasc_ndom%d_eps01_t' % ndom,)
for nnode in tentative_nnodes:
  fname = root_fnames[nnode]
  for i, precond in enumerate(preconds):
    tag = tags[i]
    iters[precond][nnode] = np.mean(np.load('data/%s.%s.pcg-iters.nreals1000.npz' % (fname, tag)))

plot_iters_vs_dofs(iters, pb='Example06')



iters2 = {precond: {indom: 0 for indom in ndoms} for precond in preconds}
for indom in ndoms:
  fname = root_fname
  tags = ('amg_0', 'amg_t',
          'bJ_nb%d_0' % indom,
          'bJ_nb%d_t' % indom,
          'lorasc_ndom%d_eps00_0' % indom,
          'lorasc_ndom%d_eps00_t' % indom,
          'lorasc_ndom%d_eps01_0' % indom,
          'lorasc_ndom%d_eps01_t' % indom,)
  for i, precond in enumerate(preconds):
    tag = tags[i]
    iters2[precond][indom] = np.mean(np.load('data/%s.%s.pcg-iters.nreals1000.npz' % (fname, tag)))

plot_iters_vs_ndoms(iters2, pb='Example06')