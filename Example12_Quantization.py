import numpy as np


distance = ('L2-full', 'cdf-full',)
Ps = (5, 10, 20, 30, 40, 50, 60, 70 ,80, 90, 100,)
nsmp_preconds = 10_000
nsmp = 1_000
root_fname = 'SExp_sig21.0_L0.1_DoF20000'

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
fig, axes = pl.subplots(2, 2, figsize=(8, 6.5), sharey='row')
axes[0, 0].set_title(distance[0])
axes[0, 0].plot(Ps, mean_iters[distance[0]])
axes[1, 0].plot(Ps, std_iters[distance[0]])

axes[0, 1].set_title(distance[1])
axes[0, 1].plot(Ps, mean_iters[distance[1]])
axes[1, 1].plot(Ps, std_iters[distance[1]])

axes[0, 0].set_ylabel('mean of PCG iterations')
axes[1, 0].set_ylabel('std of PCG iterations')

for ax in axes[1, :]:
  ax.set_xlabel('# of preconditioners')

for ax in axes.flatten(): 
  ax.grid(ls='-.')
   

pl.savefig("Example12.pdf")