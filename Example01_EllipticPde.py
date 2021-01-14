import numpy as np
import pylab as pl
from matplotlib import cm
from postproc_utils import to_axis

tentative_nnodes = 1_000_000

cells = np.load('data/DoF%d.cells.npz' % tentative_nnodes)
points = np.load('data/DoF%d.points.npz' % tentative_nnodes)
u = np.load('data/DoF%d.u.npz' % tentative_nnodes)

fig, axes = pl.subplots()
axes = to_axis(u, cells, points, axes, plot='normal', 
               vmin=2.3, vmax=3)
cbar = fig.colorbar(axes)
pl.show()