import numpy as np
import pylab as pl
from postproc_utils import to_axis, get_root_filename, get_root_filename, plot_iters_vs_dofs, plot_iters_vs_ndoms_DefPCG, plot_iters_vs_dofs_DefPCG

tentative_nnode = 32_000 # 4_000, 16_000, 32_000, 64_000, 128_000
model = 'SExp'
L = .1 # .1, .5
sig2 = 1. # .5, 1.
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

nreals = 3
ndoms = [5, 7, 10, 15, 20, 30] 

ndom = 10 # [5, 7, 10, 15, 20, 30]
tentative_nnodes = (4_000, 8_000, 16_000, 32_000, 64_000, 128_000,)
root_fnames = {nnode: get_root_filename(model, sig2, L, nnode) for nnode in tentative_nnodes}

methods = ("PCG",
           "RR-Def-PCG", "TR-RR-Def-PCG", "LO-TR-RR-Def-PCG",
           "HR-Def-PCG", "TR-HR-Def-PCG", "LO-TR-HR-Def-PCG",)

iters = {method: {indom: 0 for indom in ndoms} for method in methods}
for indom in ndoms:
  for method in methods:
    tag = 'lorasc%d_1-%s_nvec%d_spdim%d.it' % (indom, method, indom, 3*indom)
    iters[method][indom] = np.load('data/%s_%s.npz' % (root_fname, tag)).mean()
plot_iters_vs_ndoms_DefPCG(iters, precond='lorasc')
#print(iters)

iters = {method: {indom: 0 for indom in ndoms} for method in methods}
for indom in ndoms:
  for method in methods:
    tag = 'bj%d_0-%s_nvec%d_spdim%d.it' % (indom, method, indom, 3*indom)
    iters[method][indom] = np.load('data/%s_%s.npz' % (root_fname, tag)).mean()
plot_iters_vs_ndoms_DefPCG(iters, precond='bj')
#print(iters)

iters = {method: {nnode: 0 for nnode in tentative_nnodes} for method in methods}
for nnode in tentative_nnodes:
  root_fname = get_root_filename(model, sig2, L, nnode)
  for method in methods:
    tag = 'lorasc%d_1-%s_nvec%d_spdim%d.it' % (ndom, method, ndom, 3*ndom)
    iters[method][nnode] = np.load('data/%s_%s.npz' % (root_fname, tag)).mean()
plot_iters_vs_dofs_DefPCG(iters, precond='lorasc')
#print(iters)

iters = {method: {nnode: 0 for nnode in tentative_nnodes} for method in methods}
for nnode in tentative_nnodes:
  root_fname = get_root_filename(model, sig2, L, nnode)
  for method in methods:
    tag = 'bj%d_0-%s_nvec%d_spdim%d.it' % (ndom, method, ndom, 3*ndom)
    iters[method][nnode] = np.load('data/%s_%s.npz' % (root_fname, tag)).mean()
plot_iters_vs_dofs_DefPCG(iters, precond='bj')
#print(iters)



"""
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
"""