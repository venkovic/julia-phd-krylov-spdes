import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from postproc_utils import get_root_filename

plt.rcParams['text.usetex'] = True
params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}
plt.rcParams.update(params)

plt.rcParams['axes.labelsize']=22#19.
plt.rcParams['axes.titlesize']=22#19.
plt.rcParams['legend.fontsize']=22.#16.
plt.rcParams['xtick.labelsize']=22.
plt.rcParams['ytick.labelsize']=22.
plt.rcParams['legend.numpoints']=1

tentative_nnode = 4_000
model = 'SExp'
L = .1
sig2 = 1.
root_fname = get_root_filename(model, sig2, L, tentative_nnode)
nsmp = (0, 10, 100, 1000)

cells = np.load('data/DoF%d.cells.npz' % tentative_nnode)
points = np.load('data/DoF%d.points.npz' % tentative_nnode)
g = {m: np.load("data/%s.real_mcmc_%d.npz" % (root_fname, m)) for m in nsmp}


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 2.7))
cmap='jet'
triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells)
plt.set_cmap(cmap)
im = axes[0].tripcolor(triangulation, g[0], norm=colors.LogNorm(vmin=np.exp(-3.2), vmax=np.exp(3.2)))
axes[0].set_title(r"$\kappa(x,\boldsymbol{\xi}_0)$")
im = axes[1].tripcolor(triangulation, g[10], norm=colors.LogNorm(vmin=np.exp(-3.2), vmax=np.exp(3.2)))
axes[1].set_title(r"$\kappa(x,\boldsymbol{\xi}_{10})$")
im = axes[2].tripcolor(triangulation, g[100], norm=colors.LogNorm(vmin=np.exp(-3.2), vmax=np.exp(3.2)))
axes[2].set_title(r"$\kappa(x,\boldsymbol{\xi}_{100})$")
im = axes[3].tripcolor(triangulation, g[1000], norm=colors.LogNorm(vmin=np.exp(-3.2), vmax=np.exp(3.2)))
axes[3].set_title(r"$\kappa(x,\boldsymbol{\xi}_{1000})$")
for ax in axes:
  ax.axis('off')
  ax.set_aspect('equal')
plt.subplots_adjust(hspace=.05)
cb_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cb_ax, ticks=[0.1, 1, 10])
#fig.tight_layout(pad=.25)
plt.savefig("data/%s_4realizations.pdf" % root_fname, bbox_inches='tight')
plt.close()
