import numpy as np
import pylab as pl
from postproc_utils import to_axis

cells = np.load('cells.npz')
points = np.load('points.npz')
u = np.load('u.npz')

fig, axes = pl.subplots()
axes = to_axis(u, cells, points, axes, plot='normal', vmin=.3, vmax=.8)
pl.show()