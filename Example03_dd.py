import numpy as np
import pylab as pl
from postproc_utils import to_axis

cells = np.load('cells.npz')
epart = np.load('epart.npz')[:, 0]
points = np.load('points.npz')
#u = np.load('u.npz')

fig, axes = pl.subplots()
axes = to_axis(epart, cells, points, axes, at='elems')
nodes_at_interface = np.load('nodes_at_interface.npz')
pl.scatter(points[nodes_at_interface, 0], points[nodes_at_interface, 1], s=1, c='k')
pl.show()