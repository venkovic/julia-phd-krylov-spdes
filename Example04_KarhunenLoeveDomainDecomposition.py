import numpy as np
import pylab as pl
from postproc_utils import to_axis


mesh = "DoF20000"
model = "SExp_sig21.0_L0.1_DoF20000"

cells = np.load('data/%s.cells.npz' % mesh)
points = np.load('data/%s.points.npz' % mesh)
g = np.load('data/%s.greal.npz' % model)

fig, axes = pl.subplots()
axes = to_axis(g, cells, points, axes, plot='normal') # vmin=.3, vmax=.8)
pl.show()