import numpy as np
import pylab as pl
from matplotlib import cm
from postproc_utils import to_axis

cells = np.load('cells.npz')
points = np.load('points.npz')
u = np.load('u.npz')

fig, axes = pl.subplots()
axes = to_axis(u, cells, points, axes, plot='normal', 
               vmin=2.3, vmax=3)
cbar = fig.colorbar(axes)
pl.show()