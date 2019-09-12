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
    '''
    Class with the fast solver for multidimensional Fokker-Planck equation
    d r(x,t) / d t = D Nabla( r(x,t) ) - div( f(x,t) r(x,t) ), r(x,0) = r0(x),
    with known f(x,t), r0(x) and scalar coefficient D.

    Full (numpy, NP) of sparse (tensor train, TT with cross approximation)
    format may be used for the solution process.

    Basic usage:
     1 Initialize class instance with dimension, format and accuracy parameters.
     2 Call "set_grid_t" to set time grid parameters.
     3 Call "set_grid_x" to set spatial grid parameters.
     4 Call "set_funcs" to set functions in PDE and analytic solution if known.
     5 Call "set_coefs" to set coefficients in PDE.
     6 Call "prep" for initialization of the solver.
     7 Call "calc" for calculation process.
     8 Call "info" for demonstration of calculation results.
     9 Call "plot_x" for plot of solution and error at selected time moment.
    10 Call "plot_t" for plot of solution and error at selected spatial point.

    Advanced usage:
     - Call "comp" to obtain final solution at given spatial point.
    '''

    def __init__(self, d, eps=1.E-6, ord=2, with_tt=False):
        '''
        INPUT:

        d - dimension of the spatial variable
        type: int, >= 1

        eps - (optional) desired accuracy of the solution
        type: float, > 0

        ord - (optional) order of approximation
        type: int, = 1, 2
        * If = 1, then 1th order splitting and euler ode solver are used.
        * If = 2, then 2th order splitting and Runge-Kutta ode solver are used.

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
        * The min and max values are included.
        * If it is equal to 2, then the grid will be [t_min, t_max].

        t_min - (optional) min value of the time variable
        type: float, < t_max

        t_max - (optional) max value of the time variable
        type: float, > t_min

        t_hst - (optional) total number of points for history
        type: int, >= 0, <= t_poi
        * Solution at this points will be saved to history for further analysis.
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

    def set_grid_x(self, x_poi, x_min=-3., x_max=3., poi=None):
        '''
        Set parameters of the spatial grid.
        Chebyshev spatial grid is used and for each dimension the same number
        of points and limits are used.

        INPUT:

        x_poi - total number of points for each dimension
        type: int, >= 2
        * The min and max values are included.
        * If it is equal to 2, then the grid will be [x_max, x_min]^d.

        x_min - (optional) min value of the spatial variable for each dimension
        type: float, < x_max

        x_max - (optional) max value of the spatial variable for each dimension
        type: float, > x_min

        poi - (optional) special spatial point for error check
        type: ndarray (or list) [dimensions] of float or float
        default: [0, 0, ..., 0]
        * The closest point on the selected spatial grid will be used.
        * If is float, then the same value will be used for all dimensions.
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

        if poi is None: poi = np.array([0.] * self.d)
        if isinstance(poi, (int, float)): poi = np.array([float(poi)] * self.d)
        if isinstance(poi, list): poi = np.array(poi)
        self.spoi = poi

    def set_funcs(self, f0, f1, r0, rt=None, rs=None):
        '''
        Set functions for equation
        d r(x,t) / d t = D Nabla( r(x,t) ) - div( f(x,t) r(x,t) ),
        r(x,0) = r0(x), r(x, +infinity) = rs(x),
        where f0(x, t) is function f, f1(x, t) is its derivative d f / d x,
        r0(x) is initial condition, rt(x, t) is optional analytic solution
        and rs(x) is optional stationary solution.
        Functions input is X of type ndarray [dimensions, number of points].
        Functions f0 and f1 should return 2D ndarray of the same shape as X.
        Functions r0, rt and rs should return 1D ndarray of length X.shape[1].
        '''

        self.func_f0 = f0
        self.func_f1 = f1
        self.func_r0 = r0
        self.func_rt = rt
        self.func_rs = rs

    def set_coefs(self, Dc=None):
        '''
        Set coefficients for equation
        d r(x,t) / d t = D Nabla( r(x,t) ) - div( f(x,t) r(x,t) ),
        r(x,0) = r0(x), r(x, +infinity) = rs(x).

        INPUT:

        Dc - (optional) diffusion coefficient
        type: float
        default: 1.

        TODO! Replace Dc by matrix.
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

        self._t_prep = None
        self._t_calc = None
        self._t_spec = None

        self.t = self.t_min        # Current value of time variable
        self._err = None           # Current error calc vs real
        self._err_stat = None      # Current error calc vs stat
        self._err_xpoi = None      # Current error calc vs real at the sp.point
        self._err_xpoi_stat = None # Current error calc vs stat at the sp.point
        self.IT.dif2()             # Chebyshev diff. matrices
        self.D1 = self.IT.D1       # d / dx
        self.D2 = self.IT.D2       # d2 / dx2

        self.IT0 = None            # Interpolant from the previous step
        self.IT.init(self.func_r0).prep()
        #self.IT.info(0)

        J = np.eye(self.n); J[0, 0] = 0.; J[-1, -1] = 0.
        h = self.h if self.ord == 1 else self.h / 2.
        Z0 = expm(h * self.Dc * J @ self.D2)
        self.Z = Z0.copy()
        for _ in range(1, self.d):
            self.Z = np.kron(self.Z, Z0)

        self._t_prep = time.time() - _t

        self.X_hst = self.IT.grid()
        self.T_hst = []
        self.R_hst = []
        self.E_hst = []
        self.E_stat_hst = []
        self.E_xpoi_hst = []
        self.E_xpoi_stat_hst = []

        self.sind = self._sind(self.spoi)

    def calc(self):
        '''
        Calculate solution of the equation.

        TODO! Maybe move here interpolation of init. cond. from prep func.
        '''

        self._t_calc = 0.
        self._t_spec = time.time()

        _tqdm = tqdm(desc='Solve', unit='step', total=self.t_poi-1, ncols=80)
        t_hst = int(self.t_poi / self.t_hst) if self.t_hst else 0

        for i, t in enumerate(self.T):
            self.t = t
            if i == 0: # First iteration is the initial condition
                continue

            _t = time.time()
            self.IT0 = self.IT.copy()
            self.IT.init(self.step, opts={ 'is_f_with_i': self.with_tt }).prep()
            self._t_calc+= time.time() - _t
            #self.IT.info(0)

            if t_hst and (i % t_hst == 0 or i == self.t_poi - 1):
                r = self.comp()
                self.comp_real(r_calc=r)
                self.comp_stat(r_calc=r)
                self.T_hst.append(t)
                self.R_hst.append(r)

                _msg = '| At T=%-6.1e :'%self.t
                if self._err:
                    _msg+= ' e=%-6.1e'%self._err
                if self._err_stat:
                    _msg+= ' es=%-6.1e'%self._err_stat
                if not self._err and not self._err_stat:
                    _msg+= ' norm=%-6.1e'%np.linalg.norm(r)
                _tqdm.set_postfix_str(_msg, refresh=True)

            _tqdm.update(1)

        _tqdm.close()

        self.T_hst = np.array(self.T_hst)
        self.R_hst = np.array(self.R_hst)
        self.E_hst = np.array(self.E_hst)
        self.E_stat_hst = np.array(self.E_stat_hst)
        self.E_xpoi_hst = np.array(self.E_xpoi_hst)
        self.E_xpoi_stat_hst = np.array(self.E_xpoi_stat_hst)

        self._t_spec+= time.time() - self._t_spec - self._t_calc

    def step(self, X, I=None):
        '''
        One computation step for the solver.

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points] of float

        I - (optional) grid indices of the spatial variable
        type: ndarray [dimensions, number of points] of int

        OUTPUT:

        r - approximated values of the solution in the given points
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

        def step_v(X, v0, I=None):
            Z = self.Z
            if I is not None:
                I = np.ravel_multi_index(I, self.IT.n, order='F')
                Z = self.Z[np.ix_(I, I)]

            v = Z @ v0

            return v

        def step_w(X, w0):
            if self.ord == 1:
                f1 = self.func_f1(X, self.t - self.h)
                w = (1. - self.h * np.trace(f1)) * w0
            else:
                def func(y, t):
                    x = y[:-1, :]
                    r = y[-1, :]

                    f0 = self.func_f0(x, t)
                    f1 = self.func_f1(x, t)

                    return np.vstack([f0, -np.trace(f1) * r])

                y0 = np.vstack([X, w0])
                y1 = rk4(func, y0, self.t - self.h, self.t, t_poi=2)
                w = y1[-1, :]

            return w

        X0 = step_x(X)
        r0 = step_i(X0)
        v1 = step_v(X0, r0, I)
        w1 = step_w(X0, v1)
        if self.ord == 1: return w1
        v2 = step_v(X0, w1, I)
        if self.ord == 2: return v2

    def comp(self, X=None):
        '''
        Compute calculated solution at given spatial points X
        (on the history grid if is None).

        INPUT:

        X - (optional) values of the spatial variable
        type: ndarray [dimensions, number of points]

        OUTPUT:

        r - calculated solution in the given points
        type: ndarray [number of points] of float
        '''

        if X is None: X = self.X_hst

        if X is None or not X.shape[1]:
            r = None
        else:
            r = self.IT.calc(X)

        return r

    def comp_real(self, X=None, r_calc=None):
        '''
        Compute real (analytic) solution r(x, t) at given spatial points X
        (on the history grid if is None) and the corresponding error vs r_calc.
        Current time (t) is used for computation.

        INPUT:

        X - (optional) values of the spatial variable
        type: ndarray [dimensions, number of points]

        r_calc - (optional) calculated solution in the given points
        type: ndarray [number of points] of float

        OUTPUT:

        r - analytic solution in the given points
        type: ndarray [number of points] of float
        '''

        if X is None: X = self.X_hst

        if X is None or not X.shape[1] or not self.func_rt:
            r = None
            if r_calc is not None:
                self._err = None
                self._err_xpoi = None
        else:
            r = self.func_rt(X, self.t)
            if r_calc is not None:
                norm = np.linalg.norm(r)
                if norm > 0:
                    self._err = np.linalg.norm(r - r_calc) / norm
                else:
                    self._err = np.linalg.norm(r_calc)

                i = self.sind
                norm = np.abs(r[i])
                if norm > 0:
                    self._err_xpoi = np.abs(r[i] - r_calc[i]) / norm
                else:
                    self._err_xpoi = np.abs(r_calc[i])


        if r_calc is not None:
            self.E_hst.append(self._err)
            self.E_xpoi_hst.append(self._err_xpoi)

        return r

    def comp_stat(self, X=None, r_calc=None):
        '''
        Compute stationary (analytic) solution rs(x) at given spatial points X
        (on the history grid if is None) and the corresponding error vs r_calc.

        INPUT:

        X - (optional) values of the spatial variable
        type: ndarray [dimensions, number of points]

        r_calc - (optional) calculated solution in the given points
        type: ndarray [number of points] of float

        OUTPUT:

        r - stationary solution in the given points
        type: ndarray [number of points] of float
        '''

        if X is None: X = self.X_hst

        if X is None or not X.shape[1] or not self.func_rs:
            r = None
            if r_calc is not None:
                self._err_stat = None
                self._err_xpoi_stat = None
        else:
            r = self.func_rs(X)
            if r_calc is not None:
                norm = np.linalg.norm(r)
                if norm > 0:
                    self._err_stat = np.linalg.norm(r - r_calc) / norm
                else:
                    self._err_stat = np.linalg.norm(r_calc)

                i = self.sind
                norm = np.abs(r[i])
                if norm > 0:
                    self._err_xpoi_stat = np.abs(r[i] - r_calc[i]) / norm
                else:
                    self._err_xpoi_stat = np.abs(r_calc[i])

        if r_calc is not None:
            self.E_stat_hst.append(self._err_stat)
            self.E_xpoi_stat_hst.append(self._err_xpoi_stat)

        return r

    def info(self):
        '''
        Present info about the last computation.
        '''

        print('----------- Solver')
        print('Format    : %1dD, %s [order=%d]'%(self.d, 'TT, eps= %8.2e'%self.eps if self.with_tt else 'NP', self.ord))
        print('Grid x    : poi = %9d, min = %9.4f, max = %9.4f'%(self.x_poi, self.x_min, self.x_max))
        print('Grid t    : poi = %9d, min = %9.4f, max = %9.4f , hst = %9d'%(self.t_poi, self.t_min, self.t_max, self.t_hst))
        print('Time sec  : prep = %8.2e, calc = %8.2e, spec = %8.2e'%(self._t_prep, self._t_calc, self._t_spec))
        if self._err is not None:
            if self._err_xpoi is not None:
                p = ' (at the point: %8.2e)'%self._err_xpoi
            else:
                p = ''
            print('Err calc  : %8.2e%s'%(self._err, p))
        if self._err_stat is not None:
            if self._err_xpoi_stat is not None:
                p = ' (at the point: %8.2e)'%self._err_xpoi_stat
            else:
                p = ''
            print('Err stat  : %8.2e%s'%(self._err_stat, p))

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

    def plot_x(self, t=None, opts={}):
        '''
        Plot solution on the spatial grid at given time t
        (by default at the final time point):
        for the given t it finds the closest point on the time history grid.
        Initial value, analytical solution and stationary solution
        are also presented on the plot.

        INPUT:

        t - (optional) time point for plot
        type: float

        opts - (optional) dictionary with optional parameters
        type: dict with fields:

            is_log - flag:
                True  - log y-axis will be used for PDF values
                False - linear y-axis will be used for PDF values
            default: False
            type: bool

            is_abs - flag:
                True  - absolute values will be presented for PDF
                False - original values will be presented for PDF
            default: False
            type: bool

            is_err_abs - flag:
                True  - absolute error will be presented on the plot
                * err = abs(r_real(stat) - r_calc)
                False - relative error will be presented on the plot
                * err = abs(r_real(stat) - r_calc) / abs(r_real(stat))
            default: False
            type: bool

            with_err_stat - flag:
                True  - error vs stationary solution will be presented
                False - error vs stationary solution will not be presented
            default: False
            type: bool
        '''

        conf = config['opts']['plot']
        sett = config['plot']['spatial']

        i = -1 if t is None else (np.abs(self.T_hst - t)).argmin()
        t = self.T_hst[i]
        x = self.X_hst
        xg = x.reshape(-1) if self.d == 1 else np.arange(x.shape[1])

        r_init, r_stat, r_real, r_calc = None, None, None, None
        if self.func_r0: r_init = self.func_r0(x)
        if self.func_rs: r_stat = self.func_rs(x)
        if self.func_rt: r_real = self.func_rt(x, t)
        if self.R_hst is not None: r_calc = self.R_hst[i]

        e, e_stat = None, None
        if r_real is not None and r_calc is not None:
            e = np.abs(r_real - r_calc)
            if not opts.get('is_err_abs'): e/= np.abs(r_real)
        if r_stat is not None and r_calc is not None:
            if opts.get('with_err_stat'):
                e_stat = np.abs(r_stat - r_calc)
                if not opts.get('is_err_abs'): e_stat/= np.abs(r_stat)

        st = ' at t = %8.1e'%t

        fig = plt.figure(**conf['fig'][sett['fig']])
        grd = mpl.gridspec.GridSpec(**conf['grid'][sett['grid']])

        def _prep(r):
            if not isinstance(r, np.ndarray): r = np.array(r)
            return np.abs(r) if opts.get('is_abs') else r

        ax1 = fig.add_subplot(grd[0, 0])

        if r_init is not None:
            ax1.plot(xg, _prep(r_init), **{
                'label': sett['line-sol-init'][1],
                **conf['line'][sett['line-sol-init'][0]]
            })
        if r_calc is not None:
            ax1.plot(xg, _prep(r_calc), **{
                'label': sett['line-sol-calc'][1],
                **conf['line'][sett['line-sol-calc'][0]]
            })
        if r_real is not None:
            ax1.plot(xg, _prep(r_real), **{
                'label': sett['line-sol-real'][1],
                **conf['line'][sett['line-sol-real'][0]]
            })
        if r_stat is not None:
            ax1.plot(xg, _prep(r_stat), **{
                'label': sett['line-sol-stat'][1],
                **conf['line'][sett['line-sol-stat'][0]]
            })

        if opts.get('is_log'): ax1.semilogy()
        ss = ' (abs.)' if opts.get('is_abs') else ''
        ax1.set_title('%s%s%s'%(sett['title-sol'], ss, st))
        ss = ' (number)' if self.d > 1 else ' coordinate'
        ax1.set_xlabel(sett['label-sol'][0] + ss)
        ax1.set_ylabel(sett['label-sol'][1])
        ax1.legend(loc='best')

        if e is not None or e_stat is not None:
            ax2 = fig.add_subplot(grd[0, 1])

            if e is not None:
                ax2.plot(xg, e, **{
                    'label': sett['line-err-real'][1],
                    **conf['line'][sett['line-err-real'][0]]
                })
            if e_stat is not None:
                ax2.plot(xg, e_stat, **{
                    'label': sett['line-err-stat'][1],
                    **conf['line'][sett['line-err-stat'][0]]
                })

            ax2.semilogy()
            ss = ' (abs.)' if opts.get('is_err_abs') else ' (rel.)'
            ax2.set_title('%s%s%s'%(sett['title-err'], ss, st))
            ss = ' (number)' if self.d > 1 else ' coordinate'
            ax1.set_xlabel(sett['label-err'][0] + ss)
            ax2.set_ylabel(sett['label-err'][1])
            ax2.legend(loc='best')

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

    def plot_t(self, x, opts={}):
        '''
        Plot solution dependence of time at given spatial grid point x:
        for the given x it finds the closest point on the spatial grid.
        Initial value, analytical solution and stationary solution
        are also presented on the plot.

        INPUT:

        x - spatial point for plot
        type: ndarray (or list) [dimensions] of float
        * In the case of 1D it may be float.

        opts - (optional) dictionary with optional parameters
        type: dict with fields:

            is_log - flag:
                True  - log y-axis will be used for PDF values
                False - linear y-axis will be used for PDF values
            default: False
            type: bool

            is_abs - flag:
                True  - absolute values will be presented for PDF
                False - original values will be presented for PDF
            default: False
            type: bool

            is_err_abs - flag:
                True  - absolute error will be presented on the plot
                * err = abs(r_real(stat) - r_calc)
                False - relative error will be presented on the plot
                * err = abs(r_real(stat) - r_calc) / abs(r_real(stat))
            default: False
            type: bool

            with_err_stat - flag:
                True  - error vs stationary solution will be presented
                False - error vs stationary solution will not be presented
            default: False
            type: bool

        TODO! Replace full spatial grid by several selected points.
        '''

        if isinstance(x, (int, float)): x = np.array([float(x)])
        if isinstance(x, list): x = np.array(x)
        if not isinstance(x, np.ndarray) or x.shape[0] != self.d:
            s = 'Invalid spatial point.'
            raise ValueError(s)

        conf = config['opts']['plot']
        sett = config['plot']['time']

        i = self._sind(x)
        x = self.X_hst[:, i].reshape(-1, 1)
        t = self.T_hst
        v = np.ones(t.shape[0])

        r_init, r_stat, r_real, r_calc = None, None, None, None
        if self.func_r0: r_init = v * self.func_r0(x)[0]
        if self.func_rs: r_stat = v * self.func_rs(x)[0]
        if self.func_rt: r_real = np.array([self.func_rt(x, t_)[0] for t_ in t])
        if self.R_hst is not None: r_calc = np.array([r[i] for r in self.R_hst])

        e, e_stat = None, None
        if r_real is not None and r_calc is not None:
            e = np.abs(r_real - r_calc)
            if not opts.get('is_err_abs'): e/= np.abs(r_real)
            e = np.array([0.] + list(e))
        if r_stat is not None and r_calc is not None:
            if opts.get('with_err_stat'):
                e_stat = np.abs(r_stat - r_calc)
                e0 = np.abs(r_stat[0] - r_init[0])
                if opts.get('is_err_abs'):
                    e_stat = [e0] + list(e_stat)
                else:
                    e_stat/= np.abs(r_stat)
                    e_stat = [e0 / np.abs(r_stat[0])] + list(e_stat)
                e_stat = np.array(e_stat)

        sx = ', '.join(['%8.1e'%x_ for x_ in list(x.reshape(-1))])
        print('--- Solution at spatial point')
        print('X = [%s]'%sx)
        sx = ' at x = [%s]'%sx if self.d < 4 else ''

        fig = plt.figure(**conf['fig'][sett['fig']])
        grd = mpl.gridspec.GridSpec(**conf['grid'][sett['grid']])

        def _prep(r):
            if not isinstance(r, np.ndarray): r = np.array(r)
            return np.abs(r) if opts.get('is_abs') else r

        ax1 = fig.add_subplot(grd[0, 0])

        if r_init is not None:
            x_ = [self.t_min] + list(t)
            y_ = [r_init[0]] + list(r_init)
            ax1.plot(x_, _prep(y_), **{
                'label': sett['line-sol-init'][1],
                **conf['line'][sett['line-sol-init'][0]]
            })
        if r_calc is not None:
            x_ = [self.t_min] + list(t)
            y_ = [r_init[0]] + list(r_calc)
            ax1.plot(x_, _prep(y_), **{
                'label': sett['line-sol-calc'][1],
                **conf['line'][sett['line-sol-calc'][0]]
            })
        if r_real is not None:
            x_ = [self.t_min] + list(t)
            y_ = [r_init[0]] + list(r_real)
            ax1.plot(x_, _prep(y_), **{
                'label': sett['line-sol-real'][1],
                **conf['line'][sett['line-sol-real'][0]]
            })
        if r_stat is not None:
            x_ = [self.t_min] + list(t)
            y_ = [r_stat[0]] + list(r_stat)
            ax1.plot(x_, _prep(y_), **{
                'label': sett['line-sol-stat'][1],
                **conf['line'][sett['line-sol-stat'][0]]
            })

        if opts.get('is_log'): ax1.semilogy()
        ss = ' (abs.)' if opts.get('is_abs') else ''
        ax1.set_title('%s%s%s'%(sett['title-sol'], ss, sx))
        ax1.set_xlabel(sett['label-sol'][0])
        ax1.set_ylabel(sett['label-sol'][1])
        ax1.legend(loc='best')

        if e is not None or e_stat is not None:
            ax2 = fig.add_subplot(grd[0, 1])

            if e is not None:
                x_ = [self.t_min] + list(t)
                y_ = e
                ax2.plot(x_, y_, **{
                    'label': sett['line-err-real'][1],
                    **conf['line'][sett['line-err-real'][0]]
                })
            if e_stat is not None:
                x_ = [self.t_min] + list(t)
                y_ = e_stat
                ax2.plot(x_, y_, **{
                    'label': sett['line-err-stat'][1],
                    **conf['line'][sett['line-err-stat'][0]]
                })

            ax2.semilogy()
            ss = ' (abs.)' if opts.get('is_err_abs') else ' (rel.)'
            ax2.set_title('%s%s%s'%(sett['title-err'], ss, sx))
            ax2.set_xlabel(sett['label-err'][0])
            ax2.set_ylabel(sett['label-err'][1])
            ax2.legend(loc='best')

        plt.show()

    def _sind(self, x):
        '''
        Find the nearest flatten grid index for the given spatial point.

        INPUT:

        x - spatial point
        type: ndarray (or list) [dimensions] of float

        INPUT:

        i - flatten grid index
        type:  float

        TODO! Add support for calculation without explicit spatial grid.
        '''

        if isinstance(x, list): x = np.array(x)
        if not isinstance(x, np.ndarray) or x.shape[0] != self.d:
            s = 'Invalid spatial point.'
            raise ValueError(s)

        x = np.repeat(x.reshape(-1, 1), self.X_hst.shape[1], axis=1)
        i = np.linalg.norm(self.X_hst - x, axis=0).argmin()

        return i
