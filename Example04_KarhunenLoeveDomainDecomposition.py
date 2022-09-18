import numpy as np
import pylab as pl
from postproc_utils import to_axis, get_root_filename, plot_partitioned_mesh, get_root_filename, plot_local_eigvals, plot_global_eigvals

tentative_nnode = 100_000 # 4_000, 100_000
model = 'SExp'
L = .1
Ls = [.05, .1, .5, 1., 5., 10.]# [.05, .1, .5, 1., 5., 10.]
sig2 = 1.
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

ndom = 200
ndoms = [5, 10, 20, 30, 80, 200]#[5, 10, 20, 30, 80, 200]

eparts = {indom: [] for indom in ndoms}
for indom in ndoms:
  eparts[indom] = np.load("data/DoF%d-ndom%d.epart.npz" % (tentative_nnode, indom))

cells = np.load('data/DoF%d.cells.npz' % tentative_nnode)
points = np.load('data/DoF%d.points.npz' % tentative_nnode)
g = np.load('data/%s.greal.npz' % root_fname)

fig, axes, im = to_axis(g, cells, points, plot='normal')
cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
pl.savefig('img/Example04a.png', dpi=500)

plot_partitioned_mesh(points, cells, eparts)
pl.savefig('img/Example04b.png', dpi=500)

root_fname = get_root_filename(model, sig2, L, tentative_nnode)
eigvals = {indom: [] for indom in ndoms}
for indom in ndoms:
  for idom in range(indom):
    eigvals[indom] += [np.load('data/%s.kl-local-eigvals-idom%d-ndom%d.npz' % (root_fname, idom+1, indom))]
plot_local_eigvals(eigvals)
pl.savefig('img/Example04c.pdf')

eigvals = {iL: [] for iL in Ls}
for iLs in Ls:
  root_fname = get_root_filename(model, sig2, iLs, tentative_nnode)
  eigvals[iLs] += [np.load('data/%s.kl-eigvals.npz' % root_fname)]
plot_global_eigvals(eigvals)
pl.savefig('img/Example04d.pdf')