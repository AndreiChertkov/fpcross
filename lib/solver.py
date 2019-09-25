import time
import numpy as np
from numpy import kron as np_kron
from scipy.linalg import expm as sp_expm
from scipy.integrate import solve_ivp as sp_solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tqdm import tqdm

import tt

from config import config
from intertrain import Intertrain

class Solver(object):
    '''
    Class with the fast solver for multidimensional Fokker-Planck equation (FPE)
    d r(x,t) / d t = D Nabla( r(x,t) ) - div( f(x,t) r(x,t) ), r(x,0) = r0(x),
    with known f(x,t), its spatial derivative d f(x,t) / d x,
    initial condition r0(x) and scalar diffusion coefficient D.

    Full (numpy, NP) or sparse (tensor train, TT with cross approximation)
    format may be used for the solution process.

    Basic usage:
     1 Initialize class instance with dimension, format and accuracy parameters.
     2 Call "set_funcs" to set functions for FPE and analytic solution if known.
     3 Call "set_coefs" to set coefficients for FPE.
     4 Call "set_grid_t" to set the time grid parameters.
     5 Call "set_grid_x" to set the spatial grid parameters.
     6 Call "prep" for initialization of the solver.
     7 Call "calc" for calculation process.
     8 Call "comp" to obtain final solution at any given spatial point.
     9 Call "info" for demonstration of calculation results.
    10 Call "plot_x" for plot of solution and error at selected time moment.
    11 Call "plot_t" for plot of solution and error at selected spatial point.

    Notes:
     - Instead of "set_funcs" and "set_coefs" calls, equation parameters may be
       provided from an existing model while constructor or "set_model" call.
    '''

    def __init__(self, d=None, eps=1.E-6, ord=2, with_tt=False, model=None):
        '''
        INPUT:

        d - dimension of the spatial variable
        type: int, >= 1

        eps - desired accuracy of the solution
        type: float, > 0
        * Is used only for TT-rounding operations and cross approximation
        * in the TT-format (is actual only if with_tt flag is set).

        ord - order of approximation
        type: int, = 1, 2
        * If = 1, then 1th order splitting and Euler ODE solver are used.
        * If = 2, then 2th order splitting and Runge-Kutta ODE solver are used.

        with_tt - flag:
            True  - sparse (tensor train, TT) format will be used
            False - dense (numpy, NP) format will be used
        type: bool

        model - model for equation with functions and coefficients
        type: Model
        '''

        self.d = d
        self.eps = eps
        self.ord = ord
        self.with_tt = with_tt

        if model is not None:
            self.set_model(model)
        else:
            self.set_coefs()

    def set_model(self, model):
        '''
        Set functions and coefficients from existing model for equation
        d r(x,t) / d t = D Nabla( r(x,t) ) - div( f(x,t) r(x,t) ),
        r(x,0) = r0(x), r(x, +infinity) = rs(x).
        where f0(x, t) is function f, f1(x, t) is its derivative d f / d x,
        r0(x) is initial condition, rt(x, t) is optional analytic solution
        and rs(x) is optional stationary solution.

        INPUT:

        model - model for equation with functions and coefficients
        type: Model
        '''

        self.model = model
        
        self.d = model.dim()

        d0 = model.d0()
        f0 = model.f0
        f1 = model.f1
        r0 = model.r0
        rt = model.rt if model.with_rt() else None
        rs = model.rs if model.with_rs() else None
        self.set_funcs(f0, f1, r0, rt, rs)
        self.set_coefs(d0)

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

        TODO! Add support for unknown f1 function.
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

        Dc - diffusion coefficient D
        type: float
        default: 1.

        TODO! Replace Dc by matrix.
        '''

        self.Dc = Dc if Dc is not None else 1.

    def set_grid_t(self, t_poi, t_min=0., t_max=1., t_hst=0):
        '''
        Set parameters of the uniform time grid.

        INPUT:

        t_poi - total number of points
        type: int, >= 2
        * The min and max values are included.
        * If it is equal to 2, then the grid will be [t_min, t_max].

        t_min - min value of the time variable
        type: float, < t_max

        t_max - max value of the time variable
        type: float, > t_min

        t_hst - total number of points for history
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

        self.t_poi = int(t_poi)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.t_hst = int(self.t_poi * 1. / t_hst) if t_hst else 0

        self.h = (self.t_max - self.t_min) / (self.t_poi - 1)

    def set_grid_x(self, x_poi, x_min=-3., x_max=3.):
        '''
        Set parameters of the spatial grid.
        Chebyshev spatial grid is used and for each dimension the same number
        of points and limits are used.

        INPUT:

        x_poi - total number of points for each dimension
        type: int, >= 2
        * The min and max values are included.
        * If it is equal to 2, then the grid will be [x_max, x_min]^d.

        x_min - min value of the spatial variable for each dimension
        type: float, < x_max

        x_max - max value of the spatial variable for each dimension
        type: float, > x_min
        '''

        if x_poi < 2:
            s = 'Invalid number of spatial points (should be at least 2).'
            raise ValueError(s)

        if x_min >= x_max:
            s = 'Ivalid spatial limits (min should be less of max).'
            raise ValueError(s)

        self.x_poi = int(x_poi)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        n_ = np.ones(self.d, dtype='int') * self.x_poi
        l_ = np.repeat(np.array([[self.x_min, self.x_max]]), self.d, axis=0)
        self.IT = Intertrain(n=n_, l=l_, eps=self.eps, with_tt=self.with_tt)

    def prep(self):
        '''
        Init calculation parameters and prepare special matrices.

        TODO! Check usage of J matrix.
        '''

        _t = time.time()

        self.hst = {      # Saved (history) values
            'T': [],      # time moments
            'R': [],      # solution on the spatial grid
            'E_real': [], # error vs analytic (real) solution
            'E_stat': [], # error vs stationary solution
        }
        self.tms = {      # Saved times (durations)
            'prep': 0.,   # prepare special matrices
            'calc': 0.,   # calculations
            'spec': 0.,   # special operations (like error computation)
        }

        self.t = self.t_min             # Current value of time variable

        self.D0 = self.IT.dif2().copy() # Chebyshev diff. matrix d2 / dx2
        self.J0 = np.eye(self.x_poi)
        self.J0[+0, +0] = 0.
        self.J0[-1, -1] = 0.
        self.h0 = self.h if self.ord == 1 else self.h / 2.
        self.Z0 = sp_expm(self.h0 * self.Dc * self.J0 @ self.D0)

        if not self.with_tt:
            self.Z = self.Z0.copy()
            for _ in range(1, self.d): self.Z = np_kron(self.Z, self.Z0)

        self.tms['prep'] = time.time() - _t

    def calc(self):
        '''
        Calculation of the solution.
        '''

        _tqdm = tqdm(desc='Solve', unit='step', total=self.t_poi-1, ncols=80)

        self.t = self.t_min
        self.IT.init(self.func_r0)
        self.W0 = self.IT.Y.copy() # TODO! Check
        for m in range(1, self.t_poi):
            _t = time.time()

            self.t = self.t_min + m * self.h

            self.step_v()
            self.step_w()
            if self.ord == 2: self.step_v()

            self.tms['calc']+= time.time() - _t

            _t = time.time()

            if self.t_hst and (m % self.t_hst == 0 or m == self.t_poi - 1):
                _msg = self.step_check()
                _tqdm.set_postfix_str(_msg, refresh=True)
            _tqdm.update(1)

            self.tms['spec']+= time.time() - _t

        _tqdm.close()

        _t = time.time()

        self.IT.prep()

        for n in ['T', 'R', 'E_real', 'E_stat']:
            self.hst[n] = np.array(self.hst[n])

        self.tms['spec']+= time.time() - _t

    def step_v(self):
        '''
        One computation step for the diffusion term.
        Result is saved to the IT instance.
        '''

        v0 = self.IT.Y

        if self.with_tt:
            G = tt.tensor.to_list(v0)
            for i in range(self.d):
                G[i] = np.einsum('ij,kjm->kim', self.Z0, G[i])
            v = tt.tensor.from_list(G)
        else:
            v0 = v0.reshape(-1, order='F')
            v = self.Z @ v0
            v = v.reshape(self.IT.n, order='F')

        self.IT.init(Y=v)

    def step_w(self):
        '''
        One computation step for the drift term.
        Result is saved to the IT instance.

        TODO! Check IT.copy() work.
        TODO! Is it correct to add eps to prevent zero division in cross appr?
        TODO! Try interpolation of the log.
        TODO! Try initial guess in the form of the Euler solution.
        '''

        def func(y, t):
            '''
            Compute rhs for the system of convection equations
            d x / dt = f(x, t)
            d w / dt = - Tr[ d f(x, t) / d t] w.

            INPUT:

            y - combined values of the spatial variable (x) and PDF (w)
            type: ndarray [dimensions + 1, number of points] of float
            * First dimensions rows are related to x and the last row to w.

            t - time
            type: float

            OUTPUT:

            rhs - rhs of the system
            type: ndarray [dimensions + 1, number of points] of float
            '''

            x, r = y[:-1, :], y[-1, :]

            f0 = self.func_f0(x, t)
            f1 = self.func_f1(x, t)

            return np.vstack([f0, -np.trace(f1) * r])

        def step(x):
            '''
            INPUT:

            x - values of the spatial variable
            type: ndarray [dimensions, number of points] of float

            OUTPUT:

            w - approximated values of the solution in the given points
            type: ndarray [number of points] of float
            '''

            sl = Solver.ode_solve_rk4 if self.ord == 2 else Solver.ode_solve_eul
            x0 = sl(self.func_f0, x, self.t, self.t - self.h, t_poi=2)
            w0 = IT0.calc(x0)
            y0 = np.vstack([x0, w0])
            y1 = sl(func, y0, self.t - self.h, self.t, t_poi=2)
            w1 = y1[-1, :]

            if self.with_tt and np.linalg.norm(w1) < 1.E-15:
                w1+= 1.E-15 # To prevent zero division in cross appr.

            return w1

        IT0 = self.IT.copy().prep()

        self.IT.init(step, opts={
            'nswp': 200, 'kickrank': 1, 'rf': 2, 'Y0': self.W0,
        })
        self.W0 = self.IT.Y.copy()

    def step_check(self):
        '''
        Check result of the current calculation step.

        OUTPUT:

        msg - string representation of the current step for print
        type: str

        TODO! Add initial guess r to rt and maybe rs.
        '''

        def _err_calc(r_calc, r_real):
            dr = r_real - r_calc
            n0 = r_real.norm() if self.with_tt else np.linalg.norm(r_real)
            dn = dr.norm() if self.with_tt else np.linalg.norm(dr)
            return dn / n0 if n0 > 0 else dn

        r = self.IT.Y

        self.hst['T'].append(self.t)
        self.hst['R'].append(r.copy())

        msg = '| At T=%-6.1e :'%self.t

        IT = self.IT.copy(is_full=False)
        IT.eps/= 100

        if self.func_rt:
            def func_rt(x): return self.func_rt(x, self.t)
            r_real = IT.init(func_rt).Y
            self.hst['E_real'].append(_err_calc(r, r_real))

            msg+= ' er=%-6.1e'%self.hst['E_real'][-1]

        if self.func_rs:
            def func_rs(x): return self.func_rs(x)
            r_stat = IT.init(func_rs).Y
            self.hst['E_stat'].append(_err_calc(r, r_stat))

            msg+= ' es=%-6.1e'%self.hst['E_stat'][-1]

        if not self.func_rt and not self.func_rs:
            # msg+= ' norm=%-6.1e'%np.linalg.norm(r)
            pass

        return msg

    def comp(self, x):
        '''
        Compute calculated solution at given spatial points x.

        INPUT:

        x - values of the spatial variable
        type: ndarray [dimensions, number of points]

        OUTPUT:

        r - calculated solution in the given points
        type: ndarray [number of points] of float

        TODO! Add support for one-point input.
        '''

        if self.IT.A is None:
            s = 'Solution of the equation is not calculated yet. '
            s+= 'Call "prep" and "calc" functions before.'
            raise ValueError(s)

        return self.IT.calc(x)

    def info(self):
        '''
        Present info about the last computation.

        TODO! Replace prints by string construction.
        '''

        print('----------- Solver')
        print('Format    : %1dD, %s [order=%d]'%(self.d, 'TT, eps= %8.2e'%self.eps if self.with_tt else 'NP', self.ord))
        print('Grid t    : poi = %9d, min = %9.4f, max = %9.4f'%(self.t_poi, self.t_min, self.t_max))
        print('Grid x    : poi = %9d, min = %9.4f, max = %9.4f'%(self.x_poi, self.x_min, self.x_max))
        print('Time sec  : prep = %8.2e, calc = %8.2e, spec = %8.2e'%(self.tms['prep'], self.tms['calc'],self.tms['spec']))
        if len(self.hst['E_real']):
            print('Err real  : %8.2e'%self.hst['E_real'][-1])
        if len(self.hst['E_stat']):
            print('Err stat  : %8.2e'%self.hst['E_stat'][-1])

    def anim(self, ffmpeg_path, delt=50):
        '''
        Build animation for solution on the spatial grid vs time.

        INPUT:

        ffmpeg_path - path to the ffmpeg executable
        type: str

        delt - number of frames per second
        type: int, > 0
        '''

        s = 'Is draft.'
        raise NotImplementedError(s)

        def _anim_1d(ax):
            x1d = self.X[0, :self.x_poi]
            X1 = x1d

            def run(i):
                t = self.T[i]
                r = self.R[i].reshape(self.x_poi)

                ax.clear()
                ax.set_title('PDF at t=%-8.4f'%t)
                ax.set_xlabel('x')
                ax.set_ylabel('r')
                ct = ax.plot(X1, r)
                return (ct,)

            return run

        def _anim_2d(ax):
            return # DRAFT
            x1d = self.X[0, :self.x_poi]
            X1, X2 = np.meshgrid(x1d, x1d)

            def run(i):
                t = self.T[i]
                r = self.R[i].reshape((self.x_poi, self.x_poi))

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

        t - time point for plot
        type: float

        opts - dictionary with optional parameters
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

        TODO! Build full spatial grid only if plot is required.
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
            x1d = self.X[0, :self.x_poi]
            X1, X2 = np.meshgrid(x1d, x1d)

            fig = plt.figure(figsize=(10, 6))
            gs = mpl.gridspec.GridSpec(
                ncols=4, nrows=2, left=0.01, right=0.99, top=0.99, bottom=0.01,
                wspace=0.4, hspace=0.3, width_ratios=[1, 1, 1, 1], height_ratios=[10, 1]
            )

            if self.IT1:
                ax = fig.add_subplot(gs[0, :2])
                ct1 = ax.contourf(X1, X2, self.R[0].reshape((self.x_poi, self.x_poi)))
                ax.set_title('Initial PDF (t=%f)'%self.t_min)
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')

                ax = fig.add_subplot(gs[1, :2])
                cb = plt.colorbar(ct1, cax=ax, orientation='horizontal')

            if self.IT2:
                ax = fig.add_subplot(gs[0, 2:])
                ct2 = ax.contourf(X1, X2, self.R[-1].reshape((self.x_poi, self.x_poi)))
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

        opts - dictionary with optional parameters
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

        OUTPUT:

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

    @staticmethod
    def ode_solve_eul(f, y0, t_min, t_max, t_poi=2, with_y0=False):
        '''
        Solve the ordinary differential equation (ODE) d y / d t = f(y, t)
        with multiple initial conditions by standard Euiler method.

        INPUT:

        f - function that calculate the rhs of the ODE for given points
        type: function
            inp:
                y - values of the spatial variable
                type: ndarray [dimensions, number of points] of float
                t - value of the time
                type: float
            out:
                f - function values on y points for time t
                type: ndarray [dimensions, number of points] of float

        y0 - initial values of the variable
        type: ndarray (or list) [dimensions, number of points] of float

        t_min - initial time
        type: float

        t_max - final time
        type: float

        t_poi - total number of points
        type: int, >= 2
        * The min and max values are included.
        * If it is equal to 2, then points will be [t_min, t_max].

        with_y0 - flag:
            True  - the 3th argument y0 for the current points will be passed
                    to the function f
            False - without y0 argument
        type: bool

        OUTPUT:

        y - solution at the final time step
        type: ndarray [dimensions, number of points] of float
        '''

        if t_poi < 2:
            s = 'Invalid number of time points (should be at least 2).'
            raise ValueError(s)

        if not isinstance(y0, np.ndarray): y0 = np.array(y0)

        h = (t_max - t_min) / (t_poi - 1)
        t = t_min
        y = y0.copy()

        def func(y, t):
            return f(y, t, y0) if with_y0 else f(y, t)

        for _ in range(1, t_poi):
            y+= h * func(y, t)
            t+= h

        return y

    @staticmethod
    def ode_solve_rk4(f, y0, t_min, t_max, t_poi=2, with_y0=False):
        '''
        Solve the ordinary differential equation (ODE) d y / d t = f(y, t)
        with multiple initial conditions by the 4th order Runge-Kutta method.

        INPUT:

        f - function that calculate the rhs of the ODE for given points
        type: function
            inp:
                y - values of the spatial variable
                type: ndarray [dimensions, number of points] of float
                t - value of the time
                type: float
            out:
                f - function values on y points for time t
                type: ndarray [dimensions, number of points] of float

        y0 - initial values of the variable
        type: ndarray (or list) [dimensions, number of points] of float

        t_min - initial time
        type: float

        t_max - final time
        type: float

        t_poi - total number of points
        type: int, >= 2
        * The min and max values are included.
        * If it is equal to 2, then points will be [t_min, t_max].

        with_y0 - flag:
            True  - the 3th argument y0 for the current points will be passed
                    to the function f
            False - without y0 argument
        type: bool

        OUTPUT:

        y - solution at the final time step
        type: ndarray [dimensions, number of points] of float
        '''

        if t_poi < 2:
            s = 'Invalid number of time points (should be at least 2).'
            raise ValueError(s)

        if not isinstance(y0, np.ndarray): y0 = np.array(y0)

        h = (t_max - t_min) / (t_poi - 1)
        t = t_min
        y = y0.copy()

        def func(y, t):
            return f(y, t, y0) if with_y0 else f(y, t)

        for _ in range(1, t_poi):
            k1 = h * func(y, t)
            k2 = h * func(y + 0.5 * k1, t + 0.5 * h)
            k3 = h * func(y + 0.5 * k2, t + 0.5 * h)
            k4 = h * func(y + k3, t + h)
            y+= (k1 + k2 + k2 + k3 + k3 + k4) / 6.
            t+= h

        return y

    @staticmethod
    def ode_solve_ivp(f, y0, t_min, t_max, t_poi=2, with_y0=False):
        '''
        Solve the ordinary differential equation (ODE) d y / d t = f(y, t)
        with multiple initial conditions by the standard scipy solver.

        INPUT:

        f - function that calculate the rhs of the ODE for given points
        type: function
            inp:
                y - values of the spatial variable
                type: ndarray [dimensions, number of points] of float
                t - value of the time
                type: float
            out:
                f - function values on y points for time t
                type: ndarray [dimensions, number of points] of float

        y0 - initial values of the variable
        type: ndarray (or list) [dimensions, number of points] of float

        t_min - initial time
        type: float

        t_max - final time
        type: float

        t_poi - total number of points
        type: int, >= 2
        * Is not used in the current version.

        with_y0 - flag:
            True  - the 3th argument y0 for the current points will be passed
                    to the function f
            False - without y0 argument
        type: bool

        OUTPUT:

        y - solution at the final time step
        type: ndarray [dimensions, number of points] of float
        '''

        if t_poi < 2:
            s = 'Invalid number of time points (should be at least 2).'
            raise ValueError(s)

        if not isinstance(y0, np.ndarray): y0 = np.array(y0)

        y = y0.copy()

        for j in range(y.shape[1]):

            def func(t, y):
                y_ = y.reshape(-1, 1)
                if with_y0:
                    return f(y_, t, y0[:, j].reshape(-1, 1)).reshape(-1)
                return f(y_, t).reshape(-1)

            y[:, j] = sp_solve_ivp(func, [t_min, t_max], y[:, j]).y[:, -1]

        return y
