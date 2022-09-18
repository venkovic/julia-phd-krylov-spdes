import numpy as np

Ps = (10, 100, 1_000,)
distances = ('L2', 'cdf',)
nKLs = (5, 10, 20, 30, 80, 200,)

root_fname = 'SExp_sig21.0_L0.1_DoF20000'

dt_clvq, w2_clvq, dt_kmeans, w2_kmeans = {}, {}, {}, {}

for (p, P) in enumerate(Ps):
  for dist in distances:
    for nKL in nKLs:
      dt_clvq[(P, dist, nKL)] = np.load(f"data/Example13_{root_fname}_{P}_{dist}_{nKL}.dt_clvq.npz")
      w2_clvq[(P, dist, nKL)] = np.load(f"data/Example13_{root_fname}_{P}_{dist}_{nKL}.w2_clvq.npz")
      dt_kmeans[(P, dist, nKL)] = np.load(f"data/Example13_{root_fname}_{P}_{dist}_{nKL}.dt_kmeans.npz")
      w2_kmeans[(P, dist, nKL)] = np.load(f"data/Example13_{root_fname}_{P}_{dist}_{nKL}.w2_kmeans.npz")


import pylab as pl

pl.rcParams['text.usetex'] = True
params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}
pl.rcParams.update(params)
pl.rcParams['axes.labelsize']=20#19.
pl.rcParams['axes.titlesize']=20#19.
pl.rcParams['legend.fontsize']=20.#16.
pl.rcParams['xtick.labelsize']=20.
pl.rcParams['ytick.labelsize']=20.
pl.rcParams['legend.numpoints']=1#


markers = {10: '^', 100: 'o', 1000: 'v'}

fig, axes = pl.subplots(1, 2, figsize=(12.5, 6.5))
#axes[0].set_title(distance[0])
for p in Ps:
  axes[0].loglog(nKLs, [np.mean(w2_clvq[(p, 'L2', nKL)]) for nKL in nKLs], label=r'$P=%d,\;\mathrm{CLVQ}$' % p, marker=markers[p]) 
  axes[0].loglog(nKLs, [np.mean(w2_kmeans[(p, 'L2', nKL)]) for nKL in nKLs], label=r'$P=%d,\;\mathrm{k-means}$' % p, marker=markers[p]) 
  axes[1].semilogx(nKLs, [np.mean(dt_clvq[(p, 'L2', nKL)] / dt_kmeans[(p, 'L2', nKL)]) for nKL in nKLs], label=r'$P = %d$' % p, marker=markers[p])
axes[0].set_ylabel(r'$\mathbb{E}[w_2^{(n_s)}(q_2)]$')
axes[1].set_ylabel(r'$\mathbb{E}[t_{\mathrm{CLVQ}}/t_{k-\mathrm{means}}]$')
axes[1].set_yticks([i for i in range(12)])
for ax in axes:
  ax.set_xlabel('\# of KL modes')
axes[0].legend(bbox_to_anchor=(-.3, 1.3), loc='upper left', ncol=2)
axes[1].legend()
for ax in axes.flatten(): 
  ax.grid(ls='-.')
pl.savefig("img/Example13_CLVQ_vs_k-means_L2.pdf", bbox_inches='tight')

fig, axes = pl.subplots(1, 2, figsize=(12.5, 6.5))
for p in Ps:
  axes[0].loglog(nKLs, [np.mean(w2_clvq[(p, 'cdf', nKL)]) for nKL in nKLs], label=r'$P=%d,\;\mathrm{CLVQ}$' % p, marker=markers[p]) 
  axes[0].loglog(nKLs, [np.mean(w2_kmeans[(p, 'cdf', nKL)]) for nKL in nKLs], label=r'$P=%d,\;\mathrm{k-means}$' % p, marker=markers[p]) 
  axes[1].semilogx(nKLs, [np.mean(dt_clvq[(p, 'cdf', nKL)] / dt_kmeans[(p, 'cdf', nKL)]) for nKL in nKLs], label=r'$P = %d$' % p, marker=markers[p])
axes[0].set_ylabel(r'$\mathbb{E}[w_2^{(n_s)}(q_2)]$')
axes[1].set_ylabel(r'$\mathbb{E}[t_{\mathrm{CLVQ}}/t_{k-\mathrm{means}}]$')
axes[1].set_yticks([i for i in range(12)])
for ax in axes:
  ax.set_xlabel('\# of KL modes')
axes[0].legend(bbox_to_anchor=(-.3, 1.3), loc='upper left', ncol=2)
axes[1].legend()
for ax in axes.flatten(): 
  ax.grid(ls='-.')
pl.savefig("img/Example13_CLVQ_vs_k-means_cdf.pdf", bbox_inches='tight')