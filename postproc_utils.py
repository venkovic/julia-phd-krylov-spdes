import matplotlib.tri as tri
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import numpy as np

def shuffle(values, nd=None):
  if nd is None:
    nd = values.max() + 1
  domains = np.arange(nd, dtype=int)
  random.shuffle(domains)
  for idom in range(nd):
    values[values == idom] = domains[idom]
  return values

def to_axis(values, cells, points, ax, at='nodes', plot='lognorm', cmap='jet', ticks=True, labels=False,
  levels={'lognorm': (1e-3, 1e-2, 1e-1, 1, 10,),
          'symlognorm': (-10, -1, -1e-1, -1e-2, -1e-3, 1e-3, 1e-2, 1e-1, 1, 10,),},
  vmin=None, vmax=None, add_cbar=False):

  triangulation = tri.Triangulation(points[:, 0], points[:, 1], cells)
  plt.set_cmap(cmap)

  # Plot realization
  if at == 'nodes':
    if plot == 'normal':
      ax.set_aspect('equal')
      if (vmin is not None) and (vmax is not None):
        im = ax.tripcolor(triangulation, values, norm=colors.Normalize(vmin=vmin, vmax=vmax), shading='gouraud')
      else:
        im = ax.tripcolor(triangulation, values, norm=colors.Normalize(vmin=-3.2, vmax=3.2))
      ax.axis('off')
    
    elif plot == 'lognorm':
      im = ax.tricontourf(triangulation, values, levels=levels[plot], norm=colors.LogNorm(vmin=np.exp(-3.2), vmax=np.exp(3.2)))
      #im = ax.tripcolor(triangulation, values, norm=colors.LogNorm(vmin=np.exp(-3.2), vmax=np.exp(3.2)))
    
    elif plot == 'symlognorm':
      im = ax.tricontourf(triangulation, values, levels=levels[plot], norm=colors.SymLogNorm(linthresh=0.03, vmin=-np.exp(3.2), vmax=np.exp(3.2), base=10))
      #im = ax.tripcolor(triangulation, values, norm=colors.SymLogNorm(linthresh=0.03, vmin=-np.exp(3.2), vmax=np.exp(3.2), base=10))
  
  elif at == 'elems':
    plt.set_cmap(cmap='Spectral')
    im = ax.tripcolor(triangulation, facecolors=shuffle(values))

  ax.set_xlim((0, 1))
  ax.set_ylim((0, 1))

  if ticks:
    ax.set_xticks((0, .25, .5, .75, 1))
    ax.set_yticks((0, .25, .5, .75, 1))

    if labels:
      #ax.set_xticklabels((r'$0$', '', r'$0.5$', '', r'$1$'))
      #ax.set_yticklabels((r'$0$', r'$0.25$', r'$0.50$', r'$0.75$', r'$1$'))
      pass

    else:
      ax.set_xticklabels(('', '', '', '', ''))
      ax.set_yticklabels(('', '', '', '', ''))

  else:
    ax.set_xticks([])
    ax.set_yticks([])
    # Then for minor ticks:
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

  if add_cbar:
    cb_ax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

  return im


def get_root_filename(model, sig2, L, nnode):
  fname = model + "_"
  fname += "sig2%.1f" %sig2 + "_"
  fname += "L%g" % L + "_"
  return fname + "DoF%d" % nnode