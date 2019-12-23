import time
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config
from . import Grid
from . import Solver


class Check(object):
    '''
    Class for multiple computations and result representation.
    '''

    def __init__(self, fpath=None):
        self.fpath = fpath
        self.res = {}

    def set_md(self, MD):
        self.MD = MD

    def set_tm_lim(self, t_min=+0., t_max=+1.):
        self.t_min = t_min
        self.t_max = t_max

    def set_sp_lim(self, x_min=-3., x_max=+3.):
        self.x_min = x_min
        self.x_max = x_max

    def add(self, name, eps, ord, with_tt, M, N):
        self.res[name] = {
            'eps': eps,
            'ord': ord,
            'with_tt': with_tt,
            'M': [int(m) for m in list(M)],
            'N': [int(n) for n in list(N)],
        }

    def calc(self):

        def calc_many(opts):
            for m in opts['M']:
                for n in opts['N']:
                    print('----- Computation     | m = %-8d | n = %-8d'%(m, n))
                    time.sleep(1)
                    opts['%d-%d'%(m, n)] = calc_one(opts, m, n)
            time.sleep(1)

        def calc_one(opts, m, n):
            d = self.MD.d()
            MD = self.MD
            TG = Grid(d=1, n=m, l=[self.t_min, self.t_max], k='u')
            SG = Grid(d=d, n=n, l=[self.x_min, self.x_max], k='c')
            SL = Solver(TG, SG, MD, opts['eps'], opts['ord'], opts['with_tt'])
            SL.init().prep().calc()

            tms = SL.tms
            hst = SL.hst

            return {
                't_prep': tms['prep'],
                't_calc': tms['calc'],
                'e_real': hst['E_real'][-1] if len(hst['E_real']) else None,
                'e_stat': hst['E_stat'][-1] if len(hst['E_stat']) else None,
                'avrank': hst['R'][-1].erank if opts['with_tt'] else None,
            }

        for name in self.res.keys():
            print('----- Calc for solver | "%s"'%name)

            _t = time.perf_counter()

            calc_many(self.res[name])

            _t = time.perf_counter() - _t

            print('----- Done            | Time : %-8.2e sec'%_t)

    def save(self):
        data = {
            'res': self.res,
        }
        with open(self.fpath, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        with open(self.fpath, 'rb') as f:
            data = pickle.load(f)
        self.res = data['res']

    def plot(self, name, m=None, n=None, lims={}, is_stat=False):
        if self.res.get(name) is None:
            s = 'Invalid solver name "%s".'%name
            raise ValueError(s)
        if m is None and n is None:
            s = 'Both m and n arguments are None.'
            raise ValueError(s)
        if m is not None and n is not None:
            s = 'Both m and n arguments are set.'
            raise ValueError(s)

        res = self.res[name]
        conf = config['opts']['plot']
        sett = config['plot']['conv']

        if m is None:
            c, x, v = n, res['M'], [res['%d-%d'%(m, n)] for m in res['M']]
        else:
            c, x, v = m, res['N'], [res['%d-%d'%(m, n)] for n in res['N']]

        y = [v_['e_stat' if is_stat else 'e_real'] for v_ in v]
        t = [v_['t_prep'] + v_['t_calc'] for v_ in v]

        x, y, t = np.array(x), np.array(y), np.array(t)

        l0 = (lims.get(c) or lims.get('all') or [None, None])[0]
        l1 = (lims.get(c) or lims.get('all') or [None, None])[1]
        xe, ye = x[l0:l1], y[l0:l1]

        if m is None and res['ord'] == 1:
            a, b = np.polyfit(1./xe, ye, 1)
            s_lappr = '%8.1e / m'%a
            z = a / x
        if m is None and res['ord'] == 2:
            a, b = np.polyfit(1./xe**2, ye, 1)
            s_lappr = '%8.1e / m^2'%a
            z = a / x**2
        if n is None:
            b, a = np.polyfit(xe, np.log(ye), 1, w=np.sqrt(ye))
            a = np.exp(a)
            s_lappr = '%8.1e * exp[ %9.1e * n ]'%(a, b)
            z = a * np.exp(b * x)

        s_lcalc = 'Stationary solution' if is_stat else 'Analytic solution'
        s_title = ' (%d %s-points)'%(n or m, 'x' if m is None else 't')
        s_label = 'Number of %s points'%('time' if m is None else 'spatial')

        fig = plt.figure(**conf['fig'][sett['fig']])
        grd = mpl.gridspec.GridSpec(**conf['grid'][sett['grid']])

        ax = fig.add_subplot(grd[0, 0])
        ax.plot(x, y, **conf['line'][sett['line-real'][0]], label=s_lcalc)
        ax.plot(x, z, **conf['line'][sett['line-appr'][0]], label=s_lappr)
        ax.set_title(sett['title-err'] + s_title)
        ax.set_xlabel(s_label)
        ax.set_ylabel('')
        if m is None: ax.semilogx()
        ax.semilogy()
        ax.legend(loc='best')

        ax = fig.add_subplot(grd[0, 1])
        ax.plot(x, t, **conf['line'][sett['line-real'][0]])
        ax.set_title(sett['title-time'] + s_title)
        ax.set_xlabel(s_label)
        ax.set_ylabel('')
        if m is None: ax.semilogx()
        ax.semilogy()

        plt.show()

    def plot_all(self, m=None, n=None, is_stat=False):
        if m is None and n is None:
            s = 'Both m and n arguments are None.'
            raise ValueError(s)
        if m is not None and n is not None:
            s = 'Both m and n arguments are set.'
            raise ValueError(s)

        conf = config['opts']['plot']
        sett = config['plot']['conv-all']

        s_title = ' (%d %s-points)'%(n or m, 'x' if m is None else 't')
        s_label = 'Number of %s points'%('time' if m is None else 'spatial')

        fig = plt.figure(**conf['fig'][sett['fig']])
        grd = mpl.gridspec.GridSpec(**conf['grid'][sett['grid']])

        ax1 = fig.add_subplot(grd[0, 0])
        ax2 = fig.add_subplot(grd[0, 1])

        for i, name in enumerate(self.res.keys()):
            res = self.res[name]

            if m is None:
                c, x, v = n, res['M'], [res['%d-%d'%(m, n)] for m in res['M']]
            else:
                c, x, v = m, res['N'], [res['%d-%d'%(m, n)] for n in res['N']]

            y = [v_['e_stat' if is_stat else 'e_real'] for v_ in v]
            t = [v_['t_prep'] + v_['t_calc'] for v_ in v]

            x, y, t = np.array(x), np.array(y), np.array(t)

            line = conf['line'][sett['line'][0]].copy()
            line['color'] = sett['cols'][i]
            line['markerfacecolor'] = sett['cols'][i]
            line['markeredgecolor'] = sett['cols'][i]
            ax1.plot(x, y, label=name, **line)
            ax2.plot(x, t, label=name, **line)

        ax1.set_title(sett['title-err'] + s_title)
        ax2.set_title(sett['title-time'] + s_title)
        ax1.set_xlabel(s_label)
        ax2.set_xlabel(s_label)
        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.semilogy()
        ax2.semilogy()
        if m is None: ax1.semilogx()
        if m is None: ax2.semilogx()
        ax1.legend(loc='best')
        ax2.legend(loc='best')

        plt.show()
