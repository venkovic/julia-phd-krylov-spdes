import numpy as np
import pylab as pl
from postproc_utils import get_root_filename

#pl.rc('text', usetex=True)
#pl.rcParams['text.usetex'] = True
#pl.rcParams.update(params)

tentative_nnode = 50_000
model = 'SExp'
L = .1
sig2 = 1.
root_fname = get_root_filename(model, sig2, L, tentative_nnode)

it_amg = np.load('data/test01_%s_amg_0-pcg.it.npz' % root_fname)
it_lorasc = np.load("data/test01_%s_lorasc40_0-eigdefpcg_nvec45_sdpim112.it.npz" % root_fname)


fig, axes = pl.subplots()
axes.set_title('%d DoFs' % tentative_nnode)
axes.plot([s for s in range(1, len(it_amg) + 1)], it_amg, label='amg-pcg')
axes.plot([s for s in range(1, len(it_lorasc) + 1)], it_lorasc,
          label='lorasc-eigdefpcg\nndom=40, nvec=ndom+5, spdim=2.5*nvec')
axes.grid(linestyle='-.')
axes.set_xlabel(r'$s$')
axes.set_ylabel('cg iters')
pl.legend(framealpha=1)
pl.savefig('img/Example09_test01_%s.pdf' % root_fname)