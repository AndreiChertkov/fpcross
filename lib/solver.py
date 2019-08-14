import time

import numpy as np
from numpy import kron as kron
from scipy.linalg import expm as expm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tqdm import tqdm

from intertrain import Intertrain

class Solver(object):

    def __init__(self, d, eps=1.E-6, with_tt=True):
        '''
        INPUT:

        d - dimension of the spatial variable
        type: int, >= 1

        eps - (optional) desired accuracy of the solution
        type: float > 0

        with_tt - (optional) if True then the TT-format will be used
        instead of the dense (numpy) format
        type: bool
        '''

        self.d = d
        self.eps = eps
        self.with_tt = with_tt

    def set_grid_t(self, t_poi, t_min=0., t_max=1.):
        '''
        Set parameters of the uniform time grid.

        INPUT:

        t_poi - total number of points
        type: int, >= 2

        t_min - (optional) min value of the time variable
        type: float, < t_max

        t_max - (optional) max value of the time variable
        type: float, > t_min
        '''

        self.t_poi = t_poi
        self.t_min = t_min
        self.t_max = t_max

        self.m = t_poi
        self.h = (t_max - t_min) / (t_poi - 1)

        self.T = np.linspace(t_min, t_max, t_poi)

        self.t = self.t_min # Current time

    def set_grid_x(self, x_poi, x_min=-3., x_max=3.):
        '''
        Set parameters of the Chebyshev spatial grid.

        INPUT:

        x_poi - total number of points for each dimension
        type: int, >= 2

        x_min - (optional) min value of the spatial variable for each dimension
        type: float, < x_max

        x_max - (optional) max value of the spatial variable for each dimension
        type: float, > x_min
        '''

        self.x_poi = x_poi
        self.x_min = x_min
        self.x_max = x_max

        self.n = x_poi

        n_ = np.ones(self.d, dtype='int') * x_poi
        l_ = np.repeat(np.array([[x_min, x_max]]), self.d, axis=0)
        self.IT = Intertrain(n=n_, l=l_, eps=self.eps, with_tt=self.with_tt)

    def set_funcs(self, f0, f1, r0, rt=None, rs=None):
        '''
        Set functions for equation
        d r / d t = Nabla( r ) - div( f(x, t) r ), r(x, 0) = r0,
        where f0(x, t) is function f, f1(x, t) is its derivative d f / dx,
        r0(x) is initial condition, rt(x, t) is optional analytic solution
        and rs(x) is optional stationary solution.
        * Functions f0 and f1 should return value of the same shape as x
        * Functions r0, rt and rs should return 1D array of len x.shape[1]
        '''

        self.func_f0 = f0
        self.func_f1 = f1
        self.func_r0 = r0
        self.func_rt = rt
        self.func_rs = rs

    def prep(self):
        '''
        Prepare calculation parameters and interpolation of initial condition.
        '''

        self._t_prep = None
        self._t_calc = None

        self.IT0 = None # Interpolant from the previous step
        self.IT1 = None # Initial interpolant
        self.IT2 = None # Final interpolant

        self._t_prep = time.time()

        self.I = np.eye(self.n)

        self.J = np.eye(self.n)
        self.J[0, 0] = 0.
        self.J[-1, -1] = 0.

        self.D = self.IT.dif2()
        self.D = self.J@self.D
        D_expm = expm(self.D)
        self.Z = D_expm.copy()
        for d in range(self.d-1):
            self.Z = kron(self.Z, D_expm)
        self.Z = np.exp(self.h) * self.Z @ self.J

        self.IT.init(self.func_r0).prep()

        self._t_prep = (time.time() - self._t_prep)

        self.X = self.IT.grid()
        self.R = []
        self.R.append(self.IT.calc(self.X))

    def calc(self):
        '''
        Calculate solution of the equation (call prep before!).
        '''

        def normalize():
            X = np.linspace(-3., 3., 100)
            h = X[1] - X[0]
            X = (X + h/2)[:-1].reshape(1, -1)
            Y = self.IT.calc(X)
            r_int = np.sum(Y) * h
            self.IT.A = self.IT.A / r_int

        self._t_calc = 0.
        _tqdm = tqdm(desc='Solve', unit='step', total=self.t_poi-1, ncols=70)

        self.IT1 = self.IT.copy()

        for i, t in enumerate(self.T[1:]):
            _t = time.time()

            self.t = t
            self.IT0 = self.IT.copy()
            self.IT.init(self.step).prep()

            normalize()

            self._t_calc+= time.time() - _t
            #tqdm_.set_postfix_str('Error %-8.2e'%0.02, refresh=True)
            _tqdm.update(1)

            self.R.append(self.IT.calc(self.X))

        self.IT2 = self.IT.copy()

        _tqdm.close()

    def step(self, X):
        '''
        One computation step.
        '''

        f0 = self.func_f0(X, self.t)
        f1 = self.func_f1(X, self.t)

        X0 = X - self.h * f0
        I1 = X0 < -3. # TODO! replace by real multidim x_min
        I2 = X0 > +3. # TODO! replace by real multidim x_lim

        r0 = self.IT0.calc(X0)
        r0[I1.reshape(-1)] = 0.
        r0[I2.reshape(-1)] = 0.

        v = self.Z@r0

        w = (1. - self.h * np.trace(f1)) * v

        return w

    def info(self):
        '''
        Present info about the last computation.
        '''

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

    def anim(self, ffmpeg_path, delt=50):
        '''
        Build animation for solution vs time.
        '''

        def _anim_1d(ax):
            x1d = self.X[0, :self.n]
            X1 = x1d

            def run(i):
                t = self.T[i]
                r = self.R[i].reshape(self.n)

                ax.clear()
                ax.set_title('PDF at t=%-8.4f'%t)
                ax.set_xlabel('x')
                ax.set_ylabel('r')
                ct = ax.plot(X1, r)
                return (ct,)

            return run

        def _anim_2d(ax):
            x1d = self.X[0, :self.n]
            X1, X2 = np.meshgrid(x1d, x1d)

            def run(i):
                t = self.T[i]
                r = self.R[i].reshape((self.n, self.n))

                ax.clear()
                ax.set_title('Spatial distribution (t=%f)'%t)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ct = ax.contourf(X1, X2, r)
                return (ct,)

            return run

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        if self.d == 1:
            run = _anim_1d(ax)
        #elif self.d == 2:
        #    run = _anim_2d(ax)
        else:
            raise NotImplementedError('Dim %d is not supported for anim'%self.d)

        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        anim = animation.FuncAnimation(
            fig, run, frames=len(self.T), interval=delt, blit=False)

        from IPython.display import HTML
        return HTML(anim.to_html5_video())

    def plot_x(self, t=None):
        '''
        Plot solution on the spatial grid at time t (by default at final time
        point). Initial value, analytical solution and stationary solution
        are also presented on the plot.
        '''

        def _plot_1d(t):
            i = -1 if t is None else (np.abs(self.T - t)).argmin()
            t = self.T[i]

            Xg = self.IT.grid()
            x0 = Xg.reshape(-1)

            fig = plt.figure(figsize=(10, 5))
            gs = mpl.gridspec.GridSpec(
                ncols=2, nrows=1, left=0.01, right=0.99, top=0.99, bottom=0.01,
                wspace=0.2, hspace=0.1, width_ratios=[1, 1], height_ratios=[1]
            )

            ax = fig.add_subplot(gs[0, 0])

            if self.func_rs:
                r = self.func_rs(Xg)
                ax.plot(x0, r, '--', label='Stationary',
                    linewidth=3, color='magenta')
            if self.IT1:
                r = self.IT1.calc(Xg).reshape(-1)
                ax.plot(x0, r, label='Initial',
                    color='tab:blue')
            if self.IT2:
                r = self.IT2.calc(Xg).reshape(-1)
                ax.plot(x0, r, label='Calculated',
                    color='tab:green', marker='o', markersize=5, markerfacecolor='lightgreen', markeredgecolor='g')
            if self.func_rt:
                r = self.func_rt(Xg, t)
                ax.plot(x0, r, label='Analytic',
                    color='tab:orange', marker='o', markersize=5, markerfacecolor='orange', markeredgecolor='orange')

            ax.semilogy()
            ax.set_title('PDF at t=%-8.4f'%t)
            ax.set_xlabel('x')
            ax.set_ylabel('r')
            ax.legend(loc='best')

            plt.show()

        def _plot_2d(t):

            x1d = self.X[0, :self.n]
            X1, X2 = np.meshgrid(x1d, x1d)

            fig = plt.figure(figsize=(10, 6))
            gs = mpl.gridspec.GridSpec(
                ncols=4, nrows=2, left=0.01, right=0.99, top=0.99, bottom=0.01,
                wspace=0.4, hspace=0.3, width_ratios=[1, 1, 1, 1], height_ratios=[10, 1]
            )

            if self.IT1:
                ax = fig.add_subplot(gs[0, :2])
                ct1 = ax.contourf(X1, X2, self.R[0].reshape((self.n, self.n)))
                ax.set_title('Initial PDF (t=%f)'%self.t_min)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')

                ax = fig.add_subplot(gs[1, :2])
                cb = plt.colorbar(ct1, cax=ax, orientation='horizontal')

            if self.IT2:
                ax = fig.add_subplot(gs[0, 2:])
                ct2 = ax.contourf(X1, X2, self.R[-1].reshape((self.n, self.n)))
                ax.set_title('Final PDF (t=%f)'%self.t_max)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')

                ax = fig.add_subplot(gs[1, 2:])
                cb = plt.colorbar(ct2, cax=ax, orientation='horizontal')

            plt.show()

        if self.d == 1:
            return _plot_1d(t)
        #elif self.d == 2:
        #    return _plot_2d(t)
        else:
            raise NotImplementedError('Dim %d is not supported for plot'%self.d)

    def plot_t(self, x):
        '''
        Plot solution vs time at the spatial grid point x. Initial value,
        analytical solution and stationary solution are also presented.
        '''

        def _plot_1d(x):
            T = self.T
            X = self.IT.grid()
            i = (np.abs(X.reshape(-1) - x)).argmin()
            x = X.reshape(-1)[i]
            x_ = np.array([[x]])

            fig = plt.figure(figsize=(10, 5))
            gs = mpl.gridspec.GridSpec(
                ncols=2, nrows=1, left=0.01, right=0.99, top=0.99, bottom=0.01,
                wspace=0.2, hspace=0.1, width_ratios=[1, 1], height_ratios=[1]
            )

            ax = fig.add_subplot(gs[0, 0])

            if self.func_rs:
                r = np.ones(len(T)) * self.func_rs(x_)[0]
                ax.plot(T, r, '--', label='Stationary',
                    linewidth=3, color='magenta')
            if self.IT1:
                r = np.ones(len(T)) * self.IT1.calc(x_).reshape(-1)
                ax.plot(T, r, label='Initial',
                    color='tab:blue')
            if self.IT2:
                r = [r[i] for r in self.R]
                ax.plot(T, r, label='Calculated',
                    color='tab:green', marker='o', markersize=5, markerfacecolor='lightgreen', markeredgecolor='g')
            if self.func_rt:
                r = [self.func_rt(x_, t)[0] for t in T]
                ax.plot(T, r, label='Analytic',
                    color='tab:orange', marker='o', markersize=5, markerfacecolor='orange', markeredgecolor='orange')

            ax.semilogy()
            ax.set_title('PDF at x = %-8.4f'%x)
            ax.set_xlabel('t')
            ax.set_ylabel('r')
            ax.legend(loc='best')

            if self.func_rt:
                ax = fig.add_subplot(gs[0, 1])

                R1 = np.array([r[i] for r in self.R])
                R2 = np.array([self.func_rt(x_, t)[0] for t in T])
                e = np.abs(R2 - R1) / np.abs(R1)

                ax.semilogy()
                ax.plot(T, e)
                ax.set_title('Error of PDF at x = %-8.4f'%x)
                ax.set_xlabel('t')
                ax.set_ylabel('Error')

            plt.show()

        if self.d == 1:
            return _plot_1d(x)
        else:
            raise NotImplementedError('Dim %d is not supported for plot'%self.d)
