import time

import numpy as np
from numpy import kron as kron
from scipy.linalg import expm as expm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tqdm import tqdm

from config import config
from utils import rk4, eul
from intertrain import Intertrain

class Solver(object):

    def __init__(self, d, eps=1.E-6, ord=1, with_tt=False):
        '''
        INPUT:

        d - dimension of the spatial variable
        type: int, >= 1

        eps - (optional) desired accuracy of the solution
        type: float, > 0

        ord - (optional) order of approximation
        type: int, = 1, 2

        with_tt - (optional) flag:
            True  - sparse (tensor train, TT) format will be used
            False - dense (numpy, NP) format will be used
        type: bool
        '''

        self.d = d
        self.eps = eps
        self.ord = ord
        self.with_tt = with_tt

        self.set_coefs()

    def set_grid_t(self, t_poi, t_min=0., t_max=1., t_hst=0):
        '''
        Set parameters of the uniform time grid.

        INPUT:

        t_poi - total number of points
        type: int, >= 2

        t_min - (optional) min value of the time variable
        type: float, < t_max

        t_max - (optional) max value of the time variable
        type: float, > t_min

        t_hst - (optional) total number of points for history
        * solution at this pints will be saved to history for further analysis
        type: int, >= 0, <= t_poi
        '''

        if t_poi < 2:
            s = 'Invalid number of time points (should be at least 2).'
            raise ValueError(s)

        if t_min >= t_max:
            s = 'Ivalid time limits (min should be less of max).'
            raise ValueError(s)

        if t_hst > t_poi:
            s = 'Invalid number of history time points (should be <= t_poi).'
            raise ValueError(s)

        self.t_poi = t_poi
        self.t_min = t_min
        self.t_max = t_max
        self.t_hst = t_hst

        self.m = t_poi
        self.h = (t_max - t_min) / (t_poi - 1)

        self.T = np.linspace(t_min, t_max, t_poi)

    def set_grid_x(self, x_poi, x_min=-3., x_max=3.):
        '''
        Set parameters of the spatial grid.
        * For each dimension the same number of points and limits are used.
        * Chebyshev spatial grid is used.

        INPUT:

        x_poi - total number of points for each dimension
        type: int, >= 2

        x_min - (optional) min value of the spatial variable for each dimension
        type: float, < x_max

        x_max - (optional) max value of the spatial variable for each dimension
        type: float, > x_min
        '''

        if x_poi < 2:
            s = 'Invalid number of spatial points (should be at least 2).'
            raise ValueError(s)

        if x_min >= x_max:
            s = 'Ivalid spatial limits (min should be less of max).'
            raise ValueError(s)

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
        d r / d t = D Nabla( r ) - div( f(x, t) r ), r(x, 0) = r0,
        where f0(x, t) is function f, f1(x, t) is its derivative d f / d x,
        r0(x) is initial condition, rt(x, t) is optional analytic solution
        and rs(x) is optional stationary solution.
        * All functions should support input X
        * of type ndarray [d, n_pooi] of float.
        * Functions f0, f1 should return value of the same shape as X.
        * Functions r0, rt and rs should return 1D ndarray of length X.shape[1].
        '''

        self.func_f0 = f0
        self.func_f1 = f1
        self.func_r0 = r0
        self.func_rt = rt
        self.func_rs = rs

    def set_coefs(self, Dc=None):
        '''
        Set coefficients for equation
        d r / d t = D Nabla( r ) - div( f(x, t) r ), r(x, 0) = r0,

        INPUT:

        Dc - (optional) diffusion coefficient
        type: float
        default: 1.

        TODO! Replace Dc by tensor.
        '''

        self.Dc = Dc if Dc is not None else 1.

    def prep(self):
        '''
        Init calculation parameters, prepare interpolation of
        the initial condition and calculate special matrices.

        TODO! Check usage of J matrix.
        TODO! Extend J matrix to > 1D case.
        TODO! Replace X_hst by partial grid (in selected x points).
        '''

        _t = time.time()

        self._err = None
        self._t_prep = None
        self._t_calc = None
        self._t_spec = None

        self.t = self.t_min   # Current value of time variable
        self.IT.dif2()        # Chebyshev diff. matrices
        self.D1 = self.IT.D1  # d / dx
        self.D2 = self.IT.D2  # d2 / dx2

        self.IT0 = None     # Interpolant from the previous step
        self.IT.init(self.func_r0).prep()

        h = (self.Dc * self.h) ** (1./self.d)
        J = np.eye(self.n); J[0, 0] = 0.; J[-1, -1] = 0.
        D = expm(h * J @ self.D2) @ J
        self.Z = D.copy()
        for d in range(self.d-1):
            self.Z = kron(self.Z, D)

        self._t_prep = time.time() - _t

        self.X_hst = self.IT.grid()
        self.T_hst = []
        self.R_hst = []
        self.E_hst = []

    def calc(self):
        '''
        Calculate solution of the equation.

        TODO! Maybe move interpolation of init. cond. from prep func here.
        '''

        self._t_calc = 0.
        self._t_spec = time.time()

        _tqdm = tqdm(desc='Solve', unit='step', total=self.t_poi-1, ncols=80)

        for i, t in enumerate(self.T):
            self.t = t
            if i == 0: # First iteration is the initial condition
                continue

            _t = time.time()
            self.IT0 = self.IT.copy()
            self.IT.init(self.step).prep()
            self._t_calc+= time.time() - _t

            t_hst = int(self.t_poi / self.t_hst) if self.t_hst else 0
            if t_hst and (i % t_hst == 0 or i == self.t_poi - 1):
                r = self.IT.calc(self.X_hst)

                self.T_hst.append(t)
                self.R_hst.append(r)

                _msg = '| At T = %-8.2e :'%self.t

                if self.func_rt:
                    r_calc = r
                    r_real = self.func_rt(self.X_hst, t)
                    e = np.linalg.norm(r_real - r_calc) / np.linalg.norm(r_real)
                    self.E_hst.append(e)
                    self._err = e

                    _msg+= ' error = %-8.2e'%e
                else:
                    _msg+= ' norm = %-8.2e'%np.linalg.norm(r)

                _tqdm.set_postfix_str(_msg, refresh=True)

            _tqdm.update(1)

        _tqdm.close()

        self.T_hst = np.array(self.T_hst)
        self.R_hst = np.array(self.R_hst)
        self.E_hst = np.array(self.E_hst)

        self._t_spec+= time.time() - self._t_spec - self._t_calc

    def step(self, X):
        '''
        One computation step for solver.

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points]

        OUTPUT:

        r - approximated values of the solution in given points
        type: ndarray [number of points] of float
        '''

        def step_x(X):
            if self.ord == 1:
                X0 = eul(self.func_f0, X, self.t, self.t - self.h, t_poi=2)
            else:
                X0 = rk4(self.func_f0, X, self.t, self.t - self.h, t_poi=2)

            return X0

        def step_i(X):
            r0 = self.IT0.calc(X)

            return r0

        def step_w(X, w0):
            if self.ord == 1:
                f1 = self.func_f1(X, self.t - self.h)
                w = (1. - self.h * np.trace(f1)) * w0
            else:
                def func(v, t):
                    x = v[:-1, :]
                    r = v[-1, :]

                    f0 = self.func_f0(x, t)
                    f1 = self.func_f1(x, t)

                    return np.vstack([f0, -np.trace(f1) * r])

                v0 = np.vstack([X, w0])
                v1 = rk4(func, v0, self.t - self.h, self.t, t_poi=2)
                w = v1[-1, :]

            return w

        def step_v(X, v0):
            v = self.Z @ v0
            return v

        X0 = step_x(X)
        r0 = step_i(X0)
        w1 = step_w(X0, r0)
        v1 = step_v(X0, w1)
        r1 = v1.copy()

        if self.ord == 1:
            return r1

        v1 = step_v(X0, r0)
        w1 = step_w(X0, v1)
        r2 = w1.copy()

        return (r1 + r2) / 2.

    def info(self):
        '''
        Present info about the last computation.
        '''

        print('----------- Solver')
        print('Format    : %1dD, %s [order=%d]'%(self.d, 'TT, eps= %8.2e'%self.eps if self.with_tt else 'NP', self.ord))
        print('Grid x    : poi = %9d, min = %9.4f, max = %9.4f'%(self.x_poi, self.x_min, self.x_max))
        print('Grid t    : poi = %9d, min = %9.4f, max = %9.4f , hst = %9d'%(self.t_poi, self.t_min, self.t_max, self.t_hst))
        print('Time sec  : prep = %8.2e, calc = %8.2e, spec = %8.2e'%(self._t_prep, self._t_calc, self._t_spec))
        if self._err:
            print('Rel.err.  : %8.2e'%(self._err))

    def anim(self, ffmpeg_path, delt=50):
        '''
        Build animation for solution on the spatial grid vs time.

        INPUT:

        ffmpeg_path - path to the ffmpeg executable
        type: str

        delt - (optional) number of frames per second
        type: int, > 0
        '''

        s = 'Is draft.'
        raise NotImplementedError(s)

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
            return # DRAFT
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
        else:
            s = 'Dimension number %d is not supported for animation.'%self.d
            raise NotImplementedError(s)

        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        anim = animation.FuncAnimation(
            fig, run, frames=len(self.T), interval=delt, blit=False
        )

        from IPython.display import HTML
        return HTML(anim.to_html5_video())

    def plot_x(self, t=None, is_log=False, is_abs=False, is_err_abs=False):
        '''
        Plot solution on the spatial grid at time t (by default at final time
        point). Initial value, analytical solution and stationary solution
        are also presented on the plot.
        * For the given t it finds the closest point on the time history grid.

        INPUT:

        t - time point for plot
        type: float

        is_log - (optional) flag:
            True  - log y-axis will be used for PDF values
            False - simple y-axis will be used for PDF values

        is_abs - (optional) flag:
            True  - absolute values will be presented for PDF
            False - original values will be presented for PDF

        is_err_abs - (optional) flag:
            True  - absolute error will be plotted
            * err = abs(u_real - u_calc)
            False - relative error will be plotted
            * err = abs(u_real - u_calc) / abs(u_real)
        '''

        def _prep(r):
            if not isinstance(r, np.ndarray):
                r = np.array(r)
            return np.abs(r) if is_abs else r

        def _plot_1d(t):
            i = -1 if t is None else (np.abs(self.T_hst - t)).argmin()
            t = self.T_hst[i]
            X = self.X_hst
            x = X.reshape(-1)

            fig = plt.figure(**config['plot']['fig']['base_1_2'])
            grd = mpl.gridspec.GridSpec(**config['plot']['grid']['base_1_2'])
            ax1 = fig.add_subplot(grd[0, 0])
            ax2 = fig.add_subplot(grd[0, 1])

            r_init = self.func_r0(X)
            ax1.plot(x, _prep(r_init), **config['plot']['line']['init'])

            r_calc = self.R_hst[i]
            ax1.plot(x, _prep(r_calc), **config['plot']['line']['calc'])

            if self.func_rt:
                r_real = self.func_rt(X, t)

                e = np.abs(r_real - r_calc)
                if is_err_abs:
                    e = e / np.abs(r_real)

                ax1.plot(x, _prep(r_real), **config['plot']['line']['real'])
                ax2.plot(x, e, **config['plot']['line']['errs'])

            if self.func_rs:
                r_stat = self.func_rs(X)
                ax1.plot(x, _prep(r_stat), **config['plot']['line']['stat'])

            if is_log:
                ax1.semilogy()
            if is_abs:
                ax1.set_title('PDF at t=%-8.4f (abs. value)'%t)
            else:
                ax1.set_title('PDF at t=%-8.4f'%t)
            ax1.set_xlabel('x')
            ax1.set_ylabel('r')
            ax1.legend(loc='best')

            ax2.semilogy()
            if is_err_abs:
                ax2.set_title('Absolute error of PDF at t = %-8.4f'%t)
            else:
                ax2.set_title('Relative error of PDF at t = %-8.4f'%t)
            ax2.set_xlabel('x')
            ax2.set_ylabel('Error')

            plt.show()

        def _plot_2d(t):
            return # DRAFT
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
        else:
            s = 'Dimension number %d is not supported for plot.'%self.d
            raise NotImplementedError(s)

    def plot_t(self, x, is_log=False, is_abs=False, is_err_abs=False):
        '''
        Plot solution vs time at the spatial grid point x. Initial value,
        analytical solution and stationary solution are also presented.
        * For the given x it finds the closest point on the spatial grid.

        INPUT:

        x - spatial point for plot
        type: ndarray (or list) [dimensions] of float
        * In the case of 1D it may be float

        is_log - (optional) flag:
            True  - log y-axis will be used for PDF values
            False - simple y-axis will be used for PDF values

        is_abs - (optional) flag:
            True  - absolute values will be presented for PDF
            False - original values will be presented for PDF

        is_err_abs - (optional) flag:
            True  - absolute error will be plotted
            * err = abs(u_real - u_calc)
            False - relative error will be plotted
            * err = abs(u_real - u_calc) / abs(u_real)

        TODO! Replace full spatial grid by several selected points.
        '''

        def _prep(r):
            if not isinstance(r, np.ndarray):
                r = np.array(r)
            return np.abs(r) if is_abs else r

        def _plot_1d(x):
            if isinstance(x, (list, np.ndarray)): x = x[0]
            i = (np.abs(self.X_hst[0, :] - x)).argmin()
            x = self.X_hst[0, i]
            X = np.array([[x]])
            t = self.T_hst
            v = np.ones(t.shape[0])

            fig = plt.figure(**config['plot']['fig']['base_1_2'])
            grd = mpl.gridspec.GridSpec(**config['plot']['grid']['base_1_2'])
            ax1 = fig.add_subplot(grd[0, 0])
            ax2 = fig.add_subplot(grd[0, 1])

            r_init = v * self.func_r0(X)[0]
            ax1.plot(t, _prep(r_init), **config['plot']['line']['init'])

            r_calc = np.array([r[i] for r in self.R_hst])
            ax1.plot(t, _prep(r_calc), **config['plot']['line']['calc'])

            if self.func_rt:
                r_real = np.array([self.func_rt(X, t_)[0] for t_ in t])

                e = np.abs(r_real - r_calc)
                if is_err_abs:
                    e = e / np.abs(r_real)

                ax1.plot(t, _prep(r_real), **config['plot']['line']['real'])
                ax2.plot(t, e, **config['plot']['line']['errs'])

            if self.func_rs:
                r_stat = v * self.func_rs(X)[0]
                ax1.plot(t, _prep(r_stat), **config['plot']['line']['stat'])

            if is_log:
                ax1.semilogy()
            if is_abs:
                ax1.set_title('PDF at x = %-8.4f (abs. value)'%x)
            else:
                ax1.set_title('PDF at x = %-8.4f'%x)
            ax1.set_xlabel('t')
            ax1.set_ylabel('r')
            ax1.legend(loc='best')

            ax2.semilogy()
            if is_err_abs:
                ax2.set_title('Absolute error of PDF at x = %-8.4f'%x)
            else:
                ax2.set_title('Relative error of PDF at x = %-8.4f'%x)
            ax2.set_xlabel('t')
            ax2.set_ylabel('Error')

            plt.show()

        if self.d == 1:
            return _plot_1d(x)
        else:
            s = 'Dimension number %d is not supported for plot.'%self.d
            raise NotImplementedError(s)
