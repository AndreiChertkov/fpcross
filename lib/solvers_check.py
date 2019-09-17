import time
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from solver import Solver
from config import config

class SolversCheck(object):

    def __init__(self, fpath=None):
        self.fpath = fpath
        self.res = {}

    def set_grid_t(self, t_min=0., t_max=1.):
        self.t_min = t_min
        self.t_max = t_max

    def set_grid_x(self, x_min=-3., x_max=3.):
        self.x_min = x_min
        self.x_max = x_max

    def set_funcs(self, f0, f1, r0, rt=None, rs=None):
        self.func_f0 = f0
        self.func_f1 = f1
        self.func_r0 = r0
        self.func_rt = rt
        self.func_rs = rs

    def set_coefs(self, Dc=None):
        self.Dc = Dc if Dc is not None else 1.

    def add(self, name, d, eps, ord, with_tt, M, N):
        self.res[name] = {
            'd': d,
            'eps': eps,
            'ord': ord,
            'with_tt': with_tt,
            'M': list(M),
            'N': list(N),
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
            SL = Solver(opts['d'], opts['eps'], opts['ord'], opts['with_tt'])
            SL.set_grid_t(m, self.t_min, self.t_max, 1)
            SL.set_grid_x(n, self.x_min, self.x_max)
            SL.set_funcs(
                self.func_f0, self.func_f1,
                self.func_r0, self.func_rt, self.func_rs
            )
            SL.set_coefs(self.Dc)
            SL.prep()
            SL.calc()

            return {
                't_prep': SL._t_prep,
                't_calc': SL._t_calc,
                't_spec': SL._t_spec,
                'err': SL._err,
                'err_stat': SL._err_stat,
                'err_xpoi': SL._err_xpoi,
                'err_xpoi_stat': SL._err_xpoi_stat,
            }

        for name in self.res.keys():
            _t = time.time()
            print('----- Calc for solver | "%s"'%name)
            calc_many(self.res[name])
            print('----- Done            | Time : %-8.2e sec'%(time.time()-_t))

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

    def plot(self, name, m=None, n=None, lims={}, is_stat=False, is_xpoi=False):
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

        if is_stat:
            y = [v_['err_xpoi_stat' if is_xpoi else 'err_stat'] for v_ in v]
        else:
            y = [v_['err_xpoi' if is_xpoi else 'err'] for v_ in v]

        t = [v_['t_prep'] + v_['t_calc'] for v_ in v]

        x, y, t = np.array(x), np.array(y), np.array(t)

        l0 = (lims.get(c) or lims.get('all') or [None, None])[0]
        l1 = (lims.get(c) or lims.get('all') or [None, None])[1]
        xe, ye = x[l0:l1], y[l0:l1]

        if m is None and res['ord'] == 1:
            a, b = np.polyfit(1./xe, ye, 1)
            # s_appr = '%8.1e / m + %8.1e'%(a, b)
            s_appr = '%8.1e / m'%a
            z = a / x #+ b
        elif m is None and res['ord'] == 2:
            a, b = np.polyfit(1./xe**2, ye, 1)
            s_appr = '%8.1e / m^2'%a
            z = a / x**2
        else:
            b, a = np.polyfit(xe, np.log(ye), 1, w=np.sqrt(ye))
            a = np.exp(a)
            s_appr = '%8.1e * exp[ %9.1e * n ]'%(a, b)
            z = a * np.exp(b * x)

        s_calc = 'Stationary solution' if is_stat else 'Analytic solution'
        s_title = ' at point' if is_xpoi else ''
        s_title+= ' (%d %s-points)'%(n or m, 'x' if m is None else 't')
        s_label = 'Number of %s points'%('time' if m is None else 'spatial')

        fig = plt.figure(**conf['fig'][sett['fig']])
        grd = mpl.gridspec.GridSpec(**conf['grid'][sett['grid']])

        ax = fig.add_subplot(grd[0, 0])
        ax.plot(x, y, **conf['line'][sett['line-real'][0]], label=s_calc)
        ax.plot(x, z, **conf['line'][sett['line-appr'][0]], label=s_appr)
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

    def plot_all(self, m=None, n=None, is_stat=False, is_xpoi=False):
        if m is None and n is None:
            s = 'Both m and n arguments are None.'
            raise ValueError(s)
        if m is not None and n is not None:
            s = 'Both m and n arguments are set.'
            raise ValueError(s)

        conf = config['opts']['plot']
        sett = config['plot']['conv-all']

        s_calc = 'Stationary solution' if is_stat else 'Analytic solution'
        s_title = ' at point' if is_xpoi else ''
        s_title+= ' (%d %s-points)'%(n or m, 'x' if m is None else 't')
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

            if is_stat:
                y = [v_['err_xpoi_stat' if is_xpoi else 'err_stat'] for v_ in v]
            else:
                y = [v_['err_xpoi' if is_xpoi else 'err'] for v_ in v]

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
