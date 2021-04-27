import numpy as np


distance = ('L2-full', 'cdf-full',)
Ps = (5, 10, 20, 30, 40, 50, 60, 70 ,80, 90, 100,)
nsmp_preconds = 10_000
nsmp = 1_000
root_fname = 'SExp_sig21.0_L0.1_DoF4000'

iters, assignments, dists, dists_to_0 = {}, {}, {}, {}

for dist in distance:
  iters[dist] = np.zeros((nsmp, len(Ps)), dtype=int)
  assignments[dist] = np.zeros((nsmp, len(Ps)), dtype=int)
  dists[dist] = np.zeros((nsmp, len(Ps)))
  dists_to_0[dist] = np.zeros((nsmp, len(Ps)))

  for (p, P) in enumerate(Ps):
    iters[dist][:, p] = np.load(f"data/Example12_{root_fname}_{dist}_{P}_{nsmp_preconds}.iters.npz")
    assignments[dist][:, p] = np.load(f"data/Example12_{root_fname}_{dist}_{P}_{nsmp_preconds}.assignments.npz")
    dists[dist][:, p] = np.load(f"data/Example12_{root_fname}_{dist}_{P}_{nsmp_preconds}.dists.npz")
    dists_to_0[dist][:, p] = np.load(f"data/Example12_{root_fname}_{dist}_{P}_{nsmp_preconds}.dists_to_0.npz")

mean_iters, std_iters = {}, {}
for dist in distance:
  mean_iters[dist] = np.mean(iters[dist], axis=0)
  std_iters[dist] = np.std(iters[dist], axis=0)

import pylab as pl
fig, axes = pl.subplots(1, 2, sharey=True, figsize=(12, 6.5))
axes[0].set_title(distance[0])
axes[0].errorbar(Ps, mean_iters[distance[0]], yerr=std_iters[distance[0]], fmt='o')
axes[1].set_title(distance[1])
axes[1].errorbar(Ps, mean_iters[distance[1]], yerr=std_iters[distance[1]], fmt='o')
axes[0].set_ylim(50, 150)
axes[0].set_ylabel('mean iters +/- std')
for ax in axes: 
  ax.grid(ls='-.')
  ax.set_xlabel('# of preconds')

pl.savefig("Example12.pdf")