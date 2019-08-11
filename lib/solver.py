import time

import numpy as np
from numpy import kron as kron
from scipy.linalg import expm as expm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc

from intertrain import Intertrain

class Solver(object):

    def __init__(self, d, eps=1.E-6, with_tt=True):
        self.d = d
        self.eps = eps
        self.with_tt = with_tt

    def set_grid_t(self, t_poi, t_min=0., t_max=1.):
        self.t_poi = t_poi
        self.t_min = t_min
        self.t_max = t_max

        self.m = t_poi
        self.h = (t_max - t_min) / (t_poi - 1)
        self.T = np.linspace(t_min, t_max, t_poi)

    def set_grid_x(self, n, l=[-3., 3.]):
        self.n = n
        self.l = l

        n_ = np.ones(self.d, dtype='int') * n
        l_ = np.repeat(np.array(l).reshape(1, -1), self.d, axis=0)
        self.IT = Intertrain(n=n_, l=l_, eps=self.eps, with_tt=self.with_tt)

    def set_funcs(self, func_f, func_f_der, func_r0, func_r=None):
        self.func_f = func_f
        self.func_f_der = func_f_der
        self.func_r0 = func_r0
        self.func_r = func_r

    def prep(self):
        self._t_prep = None
        self._t_calc = None

        self.IT0 = None # Interpolant from previous step
        self.IT1 = None # Initial interpolant
        self.IT2 = None # Final interpolant

        self._t_prep = time.time()

        self.I = np.eye(self.n)

        self.J = np.eye(self.n)
        self.J[0, 0] = 0.
        self.J[-1, -1] = 0.

        self.D = self.IT.dif2()
        self.D = self.J@self.D
        self.Z = np.exp(self.h) * expm(self.D) @ self.J

        self.IT.init(self.func_r0).prep()

        self._t_prep = (time.time() - self._t_prep)

    def calc(self):
        self._t_calc = time.time()


        self.IT1 = self.IT.copy()

        for i, t in enumerate(self.T[:-1]):
            self.IT0 = self.IT.copy()
            self.IT.init(self.step).prep()

        self.IT2 = self.IT.copy()

        self._t_calc = (time.time() - self._t_calc)

    def step(self, X):
        f = self.func_f(X)
        g = self.func_f_der(X)

        X0 = X - self.h * f
        I1 = X0 < -3. # TODO! replace by real x_lim
        I2 = X0 > +3. # TODO! replace by real x_lim

        r0 = self.IT0.calc(X0)
        r0[I1.reshape(-1)] = 0.
        r0[I2.reshape(-1)] = 0.

        w = (1. - self.h * np.trace(g)) * r0

        v = self.Z@w

        return v

    def info(self):
        print('Info')
        print('--- Time grid')
        print('Time points : %d'%self.t_poi)
        print('Time min    : %-8.2e'%self.t_min)
        print('Time max    : %-8.2e'%self.t_max)
        print('--- Time')
        print('Prep        : %8.2e sec. '%self._t_prep)
        print('Calc        : %8.2e sec. '%self._t_calc)

        print('Info from Intertrain')
        self.IT.info()

    def plot(self):
        if self.d == 1:
            return self.plot_1d()

        raise NotImplementedError('Dim %d is not supported for plot'%self.d)

    def plot_1d(self):
        fig = plt.figure(figsize=(6, 6))
        gs = mpl.gridspec.GridSpec(
            ncols=1, nrows=1, left=0.01, right=0.99, top=0.99, bottom=0.01,
            wspace=0.4, hspace=0.3, width_ratios=[1], height_ratios=[1]
        )

        Xg = self.IT.grid()
        x0 = Xg.reshape(-1)
        r1 = self.IT1.calc(Xg).reshape(-1)
        r2 = self.IT2.calc(Xg).reshape(-1)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(x0, r1, label='Initial')
        ax.plot(x0, r2, label='Final (calc)')
        if self.func_r:
            r2_real = self.func_r(Xg, self.t_max, Xg).reshape(-1)
            ax.plot(x0, r2_real, label='Final (real)')
        ax.semilogy()
        ax.set_title('Probability density function')
        ax.set_xlabel('x')
        ax.set_ylabel('r')
        ax.legend(loc='best')

        plt.show()
