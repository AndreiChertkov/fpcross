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
                    print('----- Computation     | m = %-6d | n = %-6d'%(m, n))
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
            }

        for name in self.res.keys():
            _t = time.time()
            print('----- Calc for solver | %s'%name)
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

    def plot_x(self, name, lims={}):
        txt = '\n'
        res = self.res[name]
        M = res['M'].copy()
        N = res['N'].copy()

        conf = config['opts']['plot']
        sett = config['plot']['conv-spatial']

        fig = plt.figure(**conf['fig'][sett['fig']])
        grd = mpl.gridspec.GridSpec(**conf['grid'][sett['grid']])

        # Plot error vs analytic
        ax = fig.add_subplot(grd[0, 0])
        txt+= '\n--------------- Approximations for error vs real solution'

        for m in M:
            data = [res['%d-%d'%(m, n)] for n in N]
            x = np.array(N)
            y = np.array([d['err'] for d in data])
            ax.plot(x, y, label='m = %d'%m)

            l0 = lims.get(m, [None, None])[0]
            l1 = lims.get(m, [None, None])[1]
            xe = x[l0:l1]
            ye = y[l0:l1]
            b, a = np.polyfit(xe, np.log(ye), 1, w=np.sqrt(ye))
            a = np.exp(a)
            z = a * np.exp(b * x)
            ax.plot(x, z, '--')

            txt+= '\n m = %8d : e = %8.2e * exp[- %8.2e * n]'%(m, a, -b)

        ax.set_title(sett['title-err'])
        ax.set_xlabel(sett['label-err'][0])
        ax.set_ylabel(sett['label-err'][1])
        ax.legend(loc='best')
        ax.semilogy()

        # Plot error vs stationary
        ax = fig.add_subplot(grd[0, 1])
        txt+= '\n--------------- Approximations for error vs stat solution'

        for m in M:
            data = [res['%d-%d'%(m, n)] for n in N]
            x = np.array(N)
            y = np.array([d['err_stat'] for d in data])
            ax.plot(x, y, label='m = %d'%m)

            l0 = lims.get(m, [None, None])[0]
            l1 = lims.get(m, [None, None])[1]
            xe = x[l0:l1]
            ye = y[l0:l1]
            b, a = np.polyfit(xe, np.log(ye), 1, w=np.sqrt(ye))
            a = np.exp(a)
            z = a * np.exp(b * x)
            ax.plot(x, z, '--')

            txt+= '\n m = %8d : e = %8.2e * exp[- %8.2e * n]'%(m, a, -b)

        ax.set_title(sett['title-err-stat'])
        ax.set_xlabel(sett['label-err-stat'][0])
        ax.set_ylabel(sett['label-err-stat'][1])
        ax.legend(loc='best')
        ax.semilogy()

        # Plot time
        ax = fig.add_subplot(grd[0, 2])

        for m in M:
            data = [res['%d-%d'%(m, n)] for n in N]
            x = np.array(N)
            y = np.array([d['t_prep'] + d['t_calc'] for d in data])
            ax.plot(x, y, label='m = %d'%m)

        ax.set_title(sett['title-time'])
        ax.set_xlabel(sett['label-time'][0])
        ax.set_ylabel(sett['label-time'][1])
        ax.legend(loc='best')
        ax.semilogy()

        plt.show()

        print(txt)
