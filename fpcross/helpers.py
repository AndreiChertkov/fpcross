import os
import time
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

css = '''
    div {
      margin: 0;
      padding: 0;
    }

    .head0 {
      padding: 10px;
      display: flex;
      background-color: #dedee2;
    }
    .head0__name {
      padding-right: 5px;
      flex: 1 1 50%;
      align-self: center;
      font-size: 36px;
      font-weight: bold;
      color: #6b7e51;
    }
    .head0__note {
      padding-left: 10px;
      flex: 1 1 50%;
      align-self: flex-end;
      font-size: 16px;
      font-style: italic;
      color: #485536;
      border-left: 5px solid #8b1d1d;
    }

    .head1 {
      margin-top: 30px;
      padding: 10px;
      border-top: 5px solid #8b1d1d;
    }
    .head1__name {
      font-size: 32px;
      font-weight: bold;
      color: #485536;
    }
    .head1__note {
      padding-left: 20px;
      font-size: 20px;
      font-style: italic;
      color: #485536;
    }

    .head2 {
      margin-top: 7px;
      padding: 5px 30px 5px;
      border-left: 2px solid #42d56e;
    }
    .head2__name {
      display: inline-block;
      max-width: 70%;
      padding: 2px 10px;
      font-size: 22px;
      font-weight: bold;
      color: #145929;
      border-bottom: 2px solid #42d56e;
    }
    .head2__note {
      padding: 5px 10px 0px;
      font-size: 14px;
      font-style: italic;
      color: #485536;
    }

    .head3 {
      margin-top: 5px;
      padding: 3px 40px 3px;
      border-left: 2px dotted #485536;
    }
    .head3__name {
      display: inline-block;
      max-width: 70%;
      padding: 2px 10px;
      font-size: 20px;
      font-weight: bold;
      color: #145929;
      border-bottom: 2px dotted #485536;
    }
    .head3__note {
      padding: 2px 10px 0px;
      font-size: 12px;
      font-style: italic;
      color: #485536;
    }

    .warn {
      background-color: #fcf2f2;
      border-color: #dfb5b4;
      border-left: 5px solid #dfb5b4;
      padding: 0.5em;
    }
    .note {
      color: #42d56e;
    }
    .note::before {
      content: 'Note. ';
      font-weight: bold;
      color: #485536;
    }

    .end {
      padding: 5px;
      border-top: 5px solid #8b1d1d;
      border-bottom: 5px solid #8b1d1d;
    }
'''

def init_jupyter():
    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
    print('Start | %s |'%t)
    print('-'*37)

    from IPython.core.display import HTML
    return HTML('<style>%s</style>' %css)

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
        opts['label'] = 'Solution'
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
        opts['label'] = ''
        ax2.plot(N, [t_[1] for t_ in t], **opts)
        ax2.semilogy()
        ax2.set_title('Calculation time (sec.; tpoi=%d)'%m)
        ax2.set_xlabel('Number of spatial points')
        ax2.set_ylabel('Time, sec')

        plt.show()

def show_mnet_ords(DATA, lims):
    M = DATA['ord%d'%1]['M'].copy()

    for m in M:
        fig = plt.figure(**config['plot']['fig']['base_1_2'])
        grd = mpl.gridspec.GridSpec(**config['plot']['grid']['base_1_2'])
        ax1 = fig.add_subplot(grd[0, 0])
        ax2 = fig.add_subplot(grd[0, 1])

        for ord in [1, 2]:
            N = DATA['ord%d'%ord]['N'].copy()
            E = DATA['ord%d'%ord]['E'].copy()
            T = DATA['ord%d'%ord]['T'].copy()

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

            if ord == 1:
                opts = config['plot']['line']['real'].copy()
            else:
                opts = config['plot']['line']['calc'].copy()
            opts['label'] = 'Solution (%dth order)'%ord
            ax1.plot(x, y, **opts)

            opts = config['plot']['line']['init'].copy()
            opts['label'] = '%8.2e * exp[- %8.2e * n]'%(a, -b)
            ax1.plot(x, z1, **opts)

            if ord == 1:
                opts = config['plot']['line']['real'].copy()
            else:
                opts = config['plot']['line']['calc'].copy()
            opts['label'] = 'Solution (%dth order)'%ord
            ax2.plot(N, [t_[1] for t_ in t], **opts)

        ax1.set_title('Relative error at final time step (tpoi=%d)'%m)
        ax1.set_xlabel('Number of spatial points')
        ax1.set_ylabel('Relative error')
        ax1.legend(loc='best')
        ax1.semilogy()

        ax2.semilogy()
        ax2.set_title('Calculation time (sec.; tpoi=%d)'%m)
        ax2.set_xlabel('Number of spatial points')
        ax2.set_ylabel('Time, sec')
        ax2.legend(loc='best')

        plt.show()
