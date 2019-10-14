import time
import types
import numpy as np
from scipy.linalg import expm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tqdm import tqdm

import tt

from . import config
from . import difscheb
from . import Grid
from . import Func
from . import OrdSolver

class Solver(object):
    '''
    Class with the fast solver for multidimensional Fokker-Planck equation (FPE)
    d r(x,t) / d t = D Nabla( r(x,t) ) - div( f(x,t) r(x,t) ), r(x,0) = r0(x),
    with known f(x,t), its spatial derivative d f(x,t) / d x,
    initial condition r0(x) and scalar diffusion coefficient D.

    Full (numpy, NP) or sparse (tensor train, TT with cross approximation)
    format may be used for the solution process.

    Basic usage:
    1 Initialize class instance with grids, model, accuracy and format options.
    2 Call "init" for initialization of the solver.
    3 Call "prep" for construction of the special matrices.
    4 Call "calc" for calculation process.
    5 Call "comp" to obtain the final solution at any given spatial point.
    6 Call "info" for demonstration of the calculation results.
    7 Call "copy" to obtain new instance with the same parameters and results.

    Advanced usage:
    - Call "plot_x" for plot of solution and error at selected time moment.
    - Call "plot_t" for plot of solution and error at selected spatial point.

    PROPS:

    TG - one-dimensional uniform time grid
    type: fpcross.Grid

    SG - one- or multi-dimensional spatial Chebyshev grid
    type: fpcross.Grid

    MD - model for equation with functions and coefficients
    type: fpcross.Model

    FN - function class that keep the current solution and its interpolation
    type: fpcross.Func

    hst - dictionary with saved (history) values
    type: dict
    fld : T - time moments
        type: list of float
    fld : R - solutions of the FPE on the spatial grid
        type: list of np.ndarray or tt-tensor [number of points] of float
    fld : E_real - errors vs analytic (real) solution
        type: list of float, >= 0
    fld : E_stat - errors vs stationary solution
        type: list of float, >= 0
    * Fields T and R are lists of length t_hst.
    * Field E_real (E_stat) is a list of length t_hst if real (stationary)
    * solution is provided and is an empty list otherwise.

    tms - Saved durations of the main operations
    type: dict
    fld : prep - time spent to prepare required matrices, etc.
        type: float, >= 0
    fld : calc - time spent to perform calculations
        type: float, >= 0
    fld : spec - time spent to perform special operations (error comp., etc)
        type: float, >= 0
    '''

    def __init__(self, TG, SG, MD, eps=1.E-6, ord=2, with_tt=False):
        '''
        INPUT:

        TG - one-dimensional time grid
        type: fpcross.Grid
        * Only uniform grids are supported in the current version.

        SG - one- or multi-dimensional spatial grid
        type: fpcross.Grid
        * Only "square" grids (each dimension has the same total number of
        * points and limits) are supported in the current version.
        * Only Chebyshev grids are supported in the current version.

        MD - model for equation with functions and coefficients
        type1: fpcross.Model
        type2: fpcross.ModelBase
        * It should be instance of the Model class (type1)
        * or use ModelBase as the parent class (type2).

        eps - desired accuracy of the solution
        type: float, >= 1.E-20
        * Is used only for TT-rounding operations and cross approximation
        * in the TT-format (is actual only if with_tt flag is set).

        ord - order of the approximation
        type: int
        enum:
            - 1 - for the 1th order splitting and Euler ODE solver
            - 2 - for the 2th order splitting and Runge-Kutta ODE solver

        with_tt - flag:
            True  - sparse (tensor train, TT) format will be used
            False - dense (numpy, NP) format will be used
        type: bool

        TODO:

        - Set t_hst as parameter.

        - Check model type.
        '''

        self.TG = TG
        self.SG = SG
        self.MD = MD
        self.eps = float(eps)
        self.ord = int(ord)
        self.with_tt = bool(with_tt)

        if self.TG.d != 1 or self.TG.kind != 'u':
            raise ValueError('Invalid time grid (should be 1-dim. uniform).')

        if self.SG.kind != 'c' or not self.SG.is_square():
            raise ValueError('Invalid spatial grid (should be square Cheb.).')

        if self.eps < 1.E-20:
            raise ValueError('Invalid accuracy parameter (should be >=1.E-20).')

        if self.ord != 1 and self.ord != 2:
            raise ValueError('Invalid order parameter (should be 1 or 2).')

        self.FN = Func(self.SG, self.eps, self.with_tt)

        t_hst = 10
        self.t_hst = int(self.TG.n0 * 1. / t_hst) if t_hst else 0

        self.init()

    def init(self):
        '''
        Init main parameters of the class instance.

        TODO:

        - Maybe move some params from __init__ to this function.
        '''

        self.hst = { 'T': [], 'R': [], 'E_real': [], 'E_stat': [] }
        self.tms = { 'prep': 0., 'calc': 0., 'spec': 0. }

    def prep(self):
        '''
        Prepare special matrices.

        TODO:

        - Check usage of J matrix.
        '''

        _t = time.perf_counter()

        D0 = difscheb(self.SG, 2)[-1]
        J0 = np.eye(self.SG.n0)
        J0[+0, +0] = 0.
        J0[-1, -1] = 0.
        h0 = self.TG.h0 if self.ord == 1 else self.TG.h0 / 2.
        Dc = self.MD.Dc()
        Z0 = expm(h0 * Dc * J0 @ D0)

        self.Z0 = Z0
        if not self.with_tt:
            for i in range(self.SG.d):
                self.Z = Z0.copy() if i == 0 else np.kron(self.Z, Z0)

        self.t = self.TG.l1 # Current time

        self.tms['prep'] = time.perf_counter() - _t

    def calc(self, with_print=True):
        '''
        Calculation of the solution.

        INPUT:

        with_print - flag:
            True  - intermediate calculation results will be printed
            False - results will not be printed
        type: bool

        TODO:

        - Check if initial W0 set as r0 is correct.


        - Check if tt-round is needed for step_v.

        - Add initial guess r to rt and maybe rs in step_check.

        TODO! Check FN.copy() work.
        TODO! Is it correct to add eps to prevent zero division in cross appr?
        TODO! Try interpolation of the log.
        TODO! Try initial guess in the form of the Euler solution.
        TODO! Check various numbers of points for ODE solvers.
        '''

        def step_v():
            ''' One computation step for the diffusion term. '''

            v0 = self.FN.Y

            if self.with_tt:
                G = tt.tensor.to_list(v0)
                for i in range(self.SG.d):
                    G[i] = np.einsum('ij,kjm->kim', self.Z0, G[i])
                v = tt.tensor.from_list(G)
                v = v.round(self.eps)
            else:
                v0 = v0.reshape(-1, order='F')
                v = self.Z @ v0
                v = v.reshape(self.SG.n, order='F')

            self.FN.init(Y=v)

        def step_w():
            ''' One computation step for the drift term. '''

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

                f0 = self.MD.f0(x, t)
                f1 = self.MD.f1(x, t)

                return np.vstack([f0, -np.trace(f1) * r])

            def step(X):
                '''
                INPUT:

                X - values of the spatial variable
                type: ndarray [dimensions, number of points] of float

                OUTPUT:

                w - approximated values of the solution in the given points
                type: ndarray [number of points] of float
                '''

                TG = Grid(1, 2, [self.t, self.t - self.TG.h0], kind='u')
                SL = OrdSolver(TG, kind='eul' if self.ord == 1 else 'rk4')
                SL.init(self.MD.f0)
                X0 = SL.comp(X)
                w0 = FN.comp(X0)
                y0 = np.vstack([x0, w0])

                TG = Grid(1, 2, [self.t - self.TG.h0, self.t], kind='u')
                SL = OrdSolver(TG, kind='eul' if self.ord == 1 else 'rk4')
                SL.init(func)
                y1 = SL.comp(np.vstack([x0, w0]))
                w1 = y1[-1, :]

                if self.with_tt and np.linalg.norm(w1) < 1.E-15:
                    w1+= 1.E-15 # To prevent zero division in cross appr.

                return w1

            FN = self.FN.copy()
            FN.calc()

            self.FN.init(step, opts={
                'nswp': 200, 'kickrank': 1, 'rf': 2, 'Y0': self.W0,
            })
            self.FN.prep()

            self.W0 = self.FN.Y.copy()

        def step_f():
            '''
            Check result of the current calculation step.

            OUTPUT:

            msg - string representation of the current step for print
            type: str

            TODO:

            - Add initial guess r to rt and maybe rs.
            '''

            def _err_calc(r_calc, r_real):
                dr = r_real - r_calc
                n0 = r_real.norm() if self.with_tt else np.linalg.norm(r_real)
                dn = dr.norm() if self.with_tt else np.linalg.norm(dr)
                return dn / n0 if n0 > 0 else dn

            r = self.FN.Y

            self.hst['T'].append(self.t)
            self.hst['R'].append(r.copy())

            msg = '| At T=%-6.1e :'%self.t

            FN = self.FN.copy(is_full=False)
            FN.eps/= 100

            if self.MD.with_rt():
                def func(x): return self.MD.rt(x, self.t)
                r_real = FN.init(func).prep().Y
                self.hst['E_real'].append(_err_calc(r, r_real))

                msg+= ' er=%-6.1e'%self.hst['E_real'][-1]

            if self.MD.with_rs():
                def func(x): return self.MD.rs(x)
                r_stat = FN.init(func).prep().Y
                self.hst['E_stat'].append(_err_calc(r, r_stat))

                msg+= ' es=%-6.1e'%self.hst['E_stat'][-1]

            return msg

        M = self.TG.n0

        if with_print:
            _tqdm = tqdm(desc='Solve', unit='step', total=M-1, ncols=80)

        _t = time.perf_counter()

        self.t = self.TG.l1
        self.FN.init(self.MD.r0)
        self.FN.prep()

        self.W0 = self.FN.Y.copy()

        self.tms['calc']+= time.perf_counter() - _t

        for m in range(1, M):
            _t = time.perf_counter()

            self.t+= self.TG.h0

            step_v()
            step_w()
            if self.ord == 2: step_v()

            self.tms['calc']+= time.perf_counter() - _t

            if self.t_hst and (m % self.t_hst == 0 or m == self.TG.n0 - 1):
                _msg = step_f()
                if with_print: _tqdm.set_postfix_str(_msg, refresh=True)
            if with_print: _tqdm.update(1)

        if with_print: _tqdm.close()

        _t = time.perf_counter()

        self.FN.prep()

        self.tms['calc']+= time.perf_counter() - _t

    def comp(self, X):
        '''
        Compute calculated solution at given spatial points x.

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points]

        OUTPUT:

        r - calculated solution in the given points
        type: ndarray [number of points] of float

        TODO:

        - Add support for one-point input.
        '''

        if self.FN.A is None:
            raise ValueError('Solution of the equation is not calculated yet. Call "prep" and "calc" functions before.')

        return self.FN.comp(X)

    def comp_rhs(self, t=None, is_real=False, is_stat=False):
        '''
        Compute the right hand side of the FPE for the current solution
        (calculated, real or stationary), using Chebyshev differential matrices.
        * This function is used only for check of the models for equations.
        * Full spatial grid (and transformation of the tt-tensor to full
        format if with_tt) is used in the current version!

        INPUT:

        t - time for computation
        type1: None
        type2: float
        * If is not set (type1), then the current time will be used.

        is_real - flag:
            True  - use real solution
            False - do not use real solution
        type: bool

        is_stat - flag:
            True  - use stationary solution
            False - do not use stationary solution
        type: bool

        OUTPUT:

        e - rhs norm divided by the solution norm
        type:  float, >= 0

        TODO:

        - Add support for computation in the TT-format.
        '''

        FN = self.FN.copy()
        if is_real:
            FN.init(self.MD.rt)
            FN.prep()
        if is_stat:
            FN.init(self.MD.rs)
            FN.prep()

        t = self.t if not t else t
        x = self.SG.comp()
        f = self.MD.f0(x, t)
        Dc = self.MD.Dc()
        r = FN.Y.full() if self.with_tt else FN.Y.copy()
        r = r.reshape(-1, order='F')

        D1, D2 = difscheb(self.SG, 2)
        #J0 = np.eye(self.SG.n0); J0[0, 0] = 0.; J0[-1, -1] = 0.; J = J0.copy()
        #for _ in range(1, self.SG.d):
        #    J = np.kron(J, J0)

        rhs = 0.
        for k in range(self.SG.d):
            M = [np.eye(self.SG.n0) for _ in range(self.SG.d)]

            M[self.SG.d - 1 - k] = D1.copy()
            _D1 = M[0].copy()
            for k_ in range(1, self.SG.d):
                _D1 = np.kron(_D1, M[k_])

            M[self.SG.d - 1 - k] = D2.copy()
            _D2 = M[0].copy()
            for k_ in range(1, self.SG.d):
                _D2 = np.kron(_D2, M[k_])

            rhs-= _D1 @ (r * f[k, :])
            rhs+= _D2 @ (r * Dc)

        return np.linalg.norm(rhs) / np.linalg.norm(r)

    def info(self, is_ret=False):
        '''
        Present information about the last computation.

        INPUT:

        is_ret - flag:
            True  - return string info
            False - print string info
        type: bool

        OUTPUT:

        s - (if is_out) string with info
        type: str
        '''

        s = '------------------ Solver\n'
        s+= 'Format    : %1dD, '%self.SG.d
        s+= 'TT, eps= %8.2e '%self.eps if self.with_tt else 'NP '
        s+= '[order=%d]\n'%self.ord

        s+='Time sec  : '
        s+= 'prep = %8.2e, '%self.tms['prep']
        s+= 'calc = %8.2e\n'%self.tms['calc']

        if len(self.hst['E_real']):
            s+= 'Err real  : %8.2e\n'%self.hst['E_real'][-1]
        if len(self.hst['E_stat']):
            s+= 'Err stat  : %8.2e\n'%self.hst['E_stat'][-1]

        if not s.endswith('\n'): s+= '\n'
        if is_ret: return s
        print(s[:-1])

    def anim(self, ffmpeg_path, delt=50):
        '''
        Build animation for solution on the spatial grid vs time.

        INPUT:

        ffmpeg_path - path to the ffmpeg executable
        type: str

        delt - number of frames per second
        type: int, > 0

        TODO:

        - Finilize.
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

        if self.SG.d == 1:
            run = _anim_1d(ax)
        else:
            s = 'Dimension number %d is not supported for animation.'%self.SG.d
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

        TODO:

        - Finilize.
        '''

        s = 'Is draft.'
        raise NotImplementedError(s)

        conf = config['opts']['plot']
        sett = config['plot']['spatial']

        i = -1 if t is None else (np.abs(self.T_hst - t)).argmin()
        t = self.T_hst[i]
        x = self.X_hst
        xg = x.reshape(-1) if self.SG.d == 1 else np.arange(x.shape[1])

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
        ss = ' (number)' if self.SG.d > 1 else ' coordinate'
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
            ss = ' (number)' if self.SG.d > 1 else ' coordinate'
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

        TODO:

        - Finilize.
        '''

        s = 'Is draft.'
        raise NotImplementedError(s)

        if isinstance(x, (int, float)): x = np.array([float(x)])
        if isinstance(x, list): x = np.array(x)
        if not isinstance(x, np.ndarray) or x.shape[0] != self.SG.d:
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
        sx = ' at x = [%s]'%sx if self.SG.d < 4 else ''

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
