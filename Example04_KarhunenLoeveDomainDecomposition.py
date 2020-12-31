import numpy as np
import pylab as pl
from postproc_utils import to_axis

cells = np.load('cells.npz')
points = np.load('points.npz')
g = np.load('g.npz')

fig, axes = pl.subplots()
axes = to_axis(g, cells, points, axes, plot='normal') # vmin=.3, vmax=.8)
pl.show()