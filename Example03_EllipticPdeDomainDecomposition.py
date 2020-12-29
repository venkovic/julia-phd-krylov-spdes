import numpy as np
import pylab as pl
from postproc_utils import to_axis

cells = np.load('cells.npz')
epart = np.load('epart.npz')[:, 0]
points = np.load('points.npz')
#u = np.load('u.npz')

nd = epart.max() + 1

fig, axes = pl.subplots()
axes = to_axis(epart, cells, points, axes, at='elems')
nodes_at_interface = np.load('nodes_at_interface.npz')
nodes_inside = [np.load('nodes_inside_%d.npz' % (i+1)) for i in range(nd)]

pl.scatter(points[nodes_at_interface, 0], points[nodes_at_interface, 1], s=1, c='k')
for id in range(nd):
  pl.scatter(points[nodes_inside[id], 0], points[nodes_inside[id], 1], s=1, c=np.random.rand(3))
pl.show()