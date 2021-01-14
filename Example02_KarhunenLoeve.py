import numpy as np
import pylab as pl
from postproc_utils import to_axis, get_root_filename

tentative_nnode = 5_000
model = 'SExp'
L = .1
sig2 = 1.
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

cells = np.load('data/DoF%d.cells.npz' % tentative_nnode)
points = np.load('data/DoF%d.points.npz' % tentative_nnode)
g = np.load('data/%s.greal.npz' % root_fname)

fig, axes = pl.subplots()
axes = to_axis(g, cells, points, axes, plot='normal')
pl.show()