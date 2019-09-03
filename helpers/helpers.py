import os
import time

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from config import config

def init_jupyter():
    fpath = os.path.join(os.path.dirname(__file__), 'style.css')
    with open(fpath, 'r' ) as f:
        style = f.read()

    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
    print('Start | %s |'%t)
    print('-'*37)

    from IPython.core.display import HTML
    HTML('<style>%s</style>' %style)

def save_mnet(fpath, DATA, ord, M, N, E, T, opts={}):
    DATA['ord%d'%ord] = {
        'opts': opts,
        'M': M,
        'N': N,
        'E': E,
        'T': T,
    }

    with open(fpath, 'wb') as f:
        pickle.dump(DATA, f)

def load_mnet(fpath):
    with open(fpath, 'rb') as f:
        DATA = pickle.load(f)
    return DATA

def show_mnet_ord(DATA, ord, lims={}):
    M = DATA['ord%d'%ord]['M'].copy()
    N = DATA['ord%d'%ord]['N'].copy()
    E = DATA['ord%d'%ord]['E'].copy()
    T = DATA['ord%d'%ord]['T'].copy()

    for m in M:
        x = np.array(N)
        y = np.array(E[m])
        t = T[m].copy()

        l0 = lims.get(m, [None, None])[0]
        l1 = lims.get(m, [None, None])[1]

        xe = x[l0:l1]
        ye = y[l0:l1]

        b, a = np.polyfit(xe, np.log(ye), 1, w=np.sqrt(ye))
        a = np.exp(a)
        z1 = a * np.exp(b * x)

        fig = plt.figure(**config['plot']['fig']['base_1_2'])
        grd = mpl.gridspec.GridSpec(**config['plot']['grid']['base_1_2'])
        ax1 = fig.add_subplot(grd[0, 0])
        ax2 = fig.add_subplot(grd[0, 1])

        opts = config['plot']['line']['calc'].copy()
        opts['label'] = 'Solution (order 2)'
        ax1.plot(x, y, **opts)

        opts = config['plot']['line']['init'].copy()
        opts['label'] = '%8.2e * exp[- %8.2e * n]'%(a, -b)
        ax1.plot(x, z1, **opts)

        ax1.set_title('Relative error at final time step (tpoi=%d)'%m)
        ax1.set_xlabel('Number of spatial points')
        ax1.set_ylabel('Relative error')
        ax1.legend(loc='best')
        ax1.semilogy()

        opts = config['plot']['line']['calc'].copy()
        opts['label'] = 'Ord 1'
        ax2.plot(N, [t_[1] for t_ in t], **opts)
        ax2.semilogy()
        ax2.set_title('Calculation time (sec.; tpoi=%d)'%m)
        ax2.set_xlabel('Number of spatial points')
        ax2.set_ylabel('Time, sec')
        ax2.legend(loc='best')

        plt.show()

def show_mnet_ords(DATA, lims):
    return
