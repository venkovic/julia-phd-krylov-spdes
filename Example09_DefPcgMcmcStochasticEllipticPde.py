import numpy as np
import pylab as pl
from matplotlib.ticker import MaxNLocator
from postproc_utils import get_root_filename

#pl.rc('text', usetex=True)
#pl.rcParams['text.usetex'] = True
#pl.rcParams.update(params)

tentative_nnode = 20_000
model = 'SExp'
L = .1
sig2 = 1.
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

nsmp_max = 5

ndom = 8
nvec = int(1.25 * ndom)
spdim = 3 * ndom
nbj = ndom

it_eigdefpcg_amg = np.load("data/test01_%s_amg_0-eigdefpcg_nvec%d_spdim%d.it.npz" % (root_fname, nvec, spdim))
it_eigdefpcg_lorasc = np.load("data/test01_%s_lorasc%d_0-eigdefpcg_nvec%d_spdim%d.it.npz" % (root_fname, ndom, nvec, spdim))
it_eigdefpcg_bj = np.load("data/test01_%s_bj%d_0-eigdefpcg_nvec%d_spdim%d.it.npz" % (root_fname, nbj, nvec, spdim))
try:
  it_eigdefpcg_chol16 = np.load("data/test01_%s_chol16_0-eigdefpcg_nvec%d_spdim%d.it.npz" % (root_fname, nvec, spdim))
  do_chol16 = True
except:
  do_chol16 = False

nsmp, nchains = it_eigdefpcg_amg.shape

nsmp = min(nsmp_max, nsmp)

fig, axes = pl.subplots()

axes.set_title(f'{tentative_nnode:,} DoFs, nvec = {nvec}, spdim = {spdim}')

svals = [s for s in range(1, nsmp + 1)]

axes.errorbar(svals, np.mean(it_eigdefpcg_bj[:nsmp, :], axis=1), yerr=np.std(it_eigdefpcg_bj[:nsmp, :], axis=1),
              label='bj%d-eigdefpcg' % nbj, 
              fmt='', lw=1,
              marker='s', ms=4,
              capsize=3, capthick=1.5,
              elinewidth=1.1)
axes.errorbar(svals, np.mean(it_eigdefpcg_lorasc[:nsmp, :], axis=1), yerr=np.std(it_eigdefpcg_lorasc[:nsmp, :], axis=1),
              label='lorasc%d-eigdefpcg' % ndom,
              fmt='', lw=1,
              marker='^', ms=4,
              capsize=3, capthick=1.5,
              elinewidth=1.1)
axes.errorbar(svals, np.mean(it_eigdefpcg_amg[:nsmp, :], axis=1), yerr=np.std(it_eigdefpcg_amg[:nsmp, :], axis=1),
              label='amg-eigdefpcg',
              fmt='', lw=1,
              marker='D', ms=4,
              capsize=3, capthick=1.5,
              elinewidth=1.1)
if do_chol16:
  axes.errorbar(svals, np.mean(it_eigdefpcg_chol16[:nsmp, :], axis=1), yerr=np.std(it_eigdefpcg_chol16[:nsmp, :], axis=1),
                label='chol16-eigdefpcg',
                fmt='', lw=1,
                marker='o', ms=4,
                capsize=3, capthick=1.5,
                elinewidth=1.1)

axes.xaxis.set_major_locator(MaxNLocator(integer=True))

axes.grid(linestyle='-.')
axes.set_xlabel(r'$s$')
axes.set_ylabel('solver iterations')
if tentative_nnode == 20_000: axes.set_ylim(0, 1_400)
pl.legend(framealpha=1)
pl.savefig('img/Example09_test01_%s.pdf' % root_fname)