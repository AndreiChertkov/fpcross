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

def timer(name):
    '''
    Save time for function call.
    '''

    def timer_(f):

        def timer__(self, *args, **kwargs):

            t = time.perf_counter()
            r = f(self, *args, **kwargs)
            t = time.perf_counter() - t

            self.tms[name]+= t

            return r

        return timer__

    return timer_

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
    - Call "plot_tm" for plot of solution and error at selected spatial point.
    - Call "plot_sp" for plot of solution and error at selected time moment.

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
    fld : diff - time spent to perform calculations for diffusion term
        type: float, >= 0
    fld : conv - time spent to perform calculations for convection term
        type: float, >= 0
    fld : post - time spent to perform calculations after each step
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
        type: fpcross.Model
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

        OUTPUT:

        SL - self
        type: fpcross.Solver

        TODO:

        - Maybe move some params from __init__ to this function.
        '''

        self.hst = {'T': [], 'R': [], 'E_real': [], 'E_stat': [], 'Int': []}
        self.tms = {
            'prep': 0.,
            'calc': 0.,
            'calc_init': 0.,
            'calc_diff': 0.,
            'calc_conv': 0.,
            'calc_post': 0.,
            'calc_last': 0.,
        }

        return self

    @timer('prep')
    def prep(self):
        '''
        Prepare special matrices.

        OUTPUT:

        SL - self
        type: fpcross.Solver

        TODO:

        - Check usage of J matrix.
        '''

        self.MD.prep()

        self.D1, self.D2 = difscheb(self.SG, 2)

        D0 = self.D2
        J0 = np.eye(self.SG.n0)
        J0[+0, +0] = 0.
        J0[-1, -1] = 0.
        h0 = self.TG.h0 if self.ord == 1 else self.TG.h0 / 2.
        Z0 = expm(h0 * self.MD.D() * J0 @ D0)

        self.Z0 = Z0
        if not self.with_tt:
            for i in range(self.SG.d):
                self.Z = Z0.copy() if i == 0 else np.kron(self.Z, Z0)

        self.t = self.TG.l1 # Current time

        return self

    @timer('calc')
    def calc(self, size_max=None, rank_max=None, with_print=True):
        '''
        Calculation of the solution.

        INPUT:

        with_print - flag:
            True  - intermediate calculation results will be printed
            False - results will not be printed
        type: bool

        OUTPUT:

        SL - self
        type: fpcross.Solver

        TODO:

        - Remove computational time measurements and move it to decorators.
        '''

        if with_print:
            _tqdm = tqdm(
                desc='Solve', unit='step', total=self.TG.n0-1, ncols=80
            )

        self.step_init()

        for m in range(1, self.TG.n0):
            self.t+= self.TG.h0
            self.step_diff()
            self.step_conv()
            if self.ord == 2: self.step_diff()

            if self.t_hst and (m % self.t_hst == 0 or m == self.TG.n0 - 1):
                _msg = self.step_post()

                if with_print: _tqdm.set_postfix_str(_msg, refresh=True)

            if with_print: _tqdm.update(1)

        if with_print: _tqdm.close()

        self.step_last()

        return self

    def comp(self, X):
        '''
        Compute calculated solution at given spatial points.

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
        FN_r = self.FN.copy()
        if is_real: FN_r.init(self.MD.rt).prep()
        if is_stat: FN_r.init(self.MD.rs).prep()

        rhs = None

        if self.with_tt:  # TT-format

            for i in range(self.SG.d):
                G = tt.tensor.to_list(FN_r.Y)
                G[i] = np.einsum('ij,kjm->kim', self.D2, G[i])

                v = tt.tensor.from_list(G)
                v = v.round(self.eps)

                rhs = v.copy() if rhs is None else rhs + v

            rhs*= self.MD.D()

            for i in range(self.SG.d):

                def func(X):
                    return self.MD.f0(X, self.t if not t else t)[i, :]

                FN_f = self.FN.copy(is_init=True).init(func).prep()

                G = tt.tensor.to_list(FN_r.Y * FN_f.Y)
                G[i] = np.einsum('ij,kjm->kim', self.D1, G[i])

                v = tt.tensor.from_list(G)
                v = v.round(self.eps)

                rhs = rhs - v

            rhs = rhs.norm() / FN_r.Y.norm()


        else:            # NP-format
            t = self.t if not t else t
            x = self.SG.comp()
            f = self.MD.f0(x, t)
            D = self.MD.D()
            r = FN_r.Y.copy().reshape(-1, order='F')

            D1, D2 = difscheb(self.SG, 2)
            #J0 = np.eye(self.SG.n0); J0[0, 0] = 0.; J0[-1, -1] = 0.; J = J0.copy()
            #for _ in range(1, self.SG.d):
            #    J = np.kron(J, J0)

            rhs = 0.
            for k in range(self.SG.d):
                M = [np.eye(self.SG.n0) for _ in range(self.SG.d)]

                M[self.SG.d - 1 - k] = self.D1.copy()
                _D1 = M[0].copy()
                for k_ in range(1, self.SG.d):
                    _D1 = np.kron(_D1, M[k_])

                M[self.SG.d - 1 - k] = self.D2.copy()
                _D2 = M[0].copy()
                for k_ in range(1, self.SG.d):
                    _D2 = np.kron(_D2, M[k_])

                rhs-= _D1 @ (r * f[k, :])
                rhs+= _D2 @ (r * D)

            rhs = np.linalg.norm(rhs) / np.linalg.norm(r)

        return rhs

    @timer('calc_init')
    def step_init(self):
        '''
        Some operations before the first computation step.

        TODO:

        - Check if initial W0 set as r0 is correct.
        '''

        self.t = self.TG.l1

        self.FN.init(self.MD.r0).prep()

        self.W0 = self.FN.Y.copy()

    @timer('calc_diff')
    def step_diff(self):
        '''
        One computation step for the diffusion term.

        TODO:

        - Check if tt-round is needed at the end.
        '''

        v0 = self.FN.Y

        if self.with_tt:  # TT-format
            G = tt.tensor.to_list(v0)
            for i in range(self.SG.d):
                G[i] = np.einsum('ij,kjm->kim', self.Z0, G[i])

            v = tt.tensor.from_list(G)
            v = v.round(self.eps)

        else:            # NP-format
            v0 = v0.reshape(-1, order='F')
            v = self.Z @ v0
            v = v.reshape(self.SG.n, order='F')

        self.FN.init(Y=v)

    @timer('calc_conv')
    def step_conv(self):
        '''
        One computation step for the drift term.

        TODO:

        - Try interpolation of the log.

        - Is it correct to add eps to prevent zero division in cross?

        - Try initial guess in the form of the Euler solution?

        - Check various numbers of points for ODE solvers.
        '''

        def func(y, t):
            '''
            Compute rhs for the system of drift equations
            d x / dt = f(x, t)
            d w / dt = - Tr[ d f(x, t) / d t ] w.

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

            x = y[:-1, :]
            r = y[-1, :]

            f0 = self.MD.f0(x, t)
            f1 = self.MD.f1(x, t)

            return np.vstack([
                f0,
                -1. * np.sum(f1, axis=0) * r
            ])

        def step(X):
            '''
            INPUT:

            X - values of the spatial variable
            type: ndarray [dimensions, number of points] of float

            OUTPUT:

            w - approximated values of the solution in the given points
            type: ndarray [number of points] of float
            '''

            t0 = self.t - self.TG.h0
            t1 = self.t
            TG = Grid(1, 2, [t0, t1], kind='u')
            kd = 'eul' if self.ord == 1 else 'rk4'

            SL = OrdSolver(TG, kind=kd, is_rev=True)
            SL.init(self.MD.f0)
            X0 = SL.comp(X)
            w0 = FN.comp(X0)
            y0 = np.vstack([X0, w0])

            SL = OrdSolver(TG, kind=kd)
            SL.init(func)
            y1 = SL.comp(y0)
            w1 = y1[-1, :]

            if self.with_tt and np.linalg.norm(w1) < 1.E-15:
                w1+= 1.E-15 # To prevent zero division in the cross appr.

            return w1

        FN = self.FN.copy()
        FN.calc()

        self.FN.init(step, opts={
            'nswp': 200, 'kickrank': 1, 'rf': 2, 'Y0': self.W0,
        })
        self.FN.prep()

        self.W0 = self.FN.Y.copy()

    @timer('calc_post')
    def step_post(self):
        '''
        Check result of the current computation step.

        TODO:

        - Add initial guess r to rt and maybe rs.
        '''

        self.FN.calc()
        nrm = self.FN.comp_int()
        # self.FN.Y = 1./nrm * self.FN.Y

        def _err_calc(r_calc, r_real):
            dr = r_real - r_calc
            n0 = r_real.norm() if self.with_tt else np.linalg.norm(r_real)
            dn = dr.norm() if self.with_tt else np.linalg.norm(dr)
            return dn / n0 if n0 > 0 else dn

        self.hst['T'].append(self.t)
        self.hst['R'].append(self.FN.Y.copy())
        self.hst['Int'].append(nrm)

        msg = '| At T=%-6.1e :'%self.t

        FN = self.FN.copy(is_init=True)
        FN.eps/= 100

        if self.MD.with_rt():
            FN.init(lambda x: self.MD.rt(x, self.t))
            FN.prep()
            self.hst['E_real'].append(_err_calc(self.FN.Y, FN.Y))

            msg+= ' er=%-6.1e'%self.hst['E_real'][-1]

        if self.MD.with_rs():
            FN.init(lambda x: self.MD.rs(x))
            FN.prep()
            self.hst['E_stat'].append(_err_calc(self.FN.Y, FN.Y))

            msg+= ' es=%-6.1e'%self.hst['E_stat'][-1]

        if self.with_tt:
            msg+= ' r=%-6.2e'%self.FN.Y.erank

        msg+= ' n=%-6.2e'%nrm

        rhs = self.comp_rhs()
        msg+= ' rhs=%-6.2e'%rhs

        return msg

    @timer('calc_last')
    def step_last(self):
        '''
        Some operations after the final computation step.
        '''

        self.FN.calc()
        # nrm = self.FN.comp_int()
        # self.FN.Y = 1./nrm * self.FN.Y
        return

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

        if len(self.hst['E_real']):
            s+= 'Err real  : %8.2e\n'%self.hst['E_real'][-1]
        if len(self.hst['E_stat']):
            s+= 'Err stat  : %8.2e\n'%self.hst['E_stat'][-1]

        s+= 'Time prep : %8.2e \n'%self.tms['prep']
        s+= 'Time calc : %8.2e \n'%self.tms['calc']
        s+= '...  init : %8.2e \n'%self.tms['calc_init']
        s+= '...  diff : %8.2e \n'%self.tms['calc_diff']
        s+= '...  conv : %8.2e \n'%self.tms['calc_conv']
        s+= '...  post : %8.2e \n'%self.tms['calc_post']
        s+= '...  last : %8.2e \n'%self.tms['calc_last']

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

    def plot_tm(self, x, opts={}):
        '''
        Plot solution dependence of time at given spatial grid point x: for the
        given x it finds the closest point on the spatial grid. Initial value,
        analytical and stationary solution are also presented on the plot.

        INPUT:

        x - spatial point for plot
        type1: float
        type2: list [dimensions] of float
        type3: ndarray [dimensions] of float
        * In the case of 1D it may be float (type1).

        opts - dictionary with optional parameters
        type: dict
        fld : is_log - flag:
                True  - log y-axis will be used for PDF values
                False - linear y-axis will be used for PDF values
            default: False
            type: bool
        fld : is_abs - flag:
                True  - absolute values will be presented for PDF
                False - original values will be presented for PDF
            default: False
            type: bool
        fld : is_err_abs - flag:
                True  - absolute error will be presented on the plot
                        ( err = abs(r_real(stat) - r_calc) )
                False - relative error will be presented on the plot
                        ( err = abs(r_real(stat) - r_calc) / abs(r_real(stat)) )
            default: False
            type: bool
        fld : with_err_stat - flag:
                True  - error vs stationary solution will be presented
                False - error vs stationary solution will not be presented
            default: False
            type: bool

        TODO:

        - Finilize.
        '''

        raise NotImplementedError('Is draft.')

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

    def plot_sp(self, t=None, opts={}):
        '''
        Plot solution on the spatial grid at given time t: for the given t it
        finds the closest point on the time history grid. Initial value,
        analytical and stationary solution are also presented on the plot.

        INPUT:

        t - time point for plot
        type1: None
        type2: float
        * If is not set (type1), then the final time point will be used.

        opts - dictionary with optional parameters
        type: dict
        fld : is_log - flag:
                True  - log y-axis will be used for PDF values
                False - linear y-axis will be used for PDF values
            default: False
            type: bool
        fld : is_abs - flag:
                True  - absolute values will be presented for PDF
                False - original values will be presented for PDF
            default: False
            type: bool
        fld : is_err_abs - flag:
                True  - absolute error will be presented on the plot
                        ( err = abs(r_real(stat) - r_calc) )
                False - relative error will be presented on the plot
                        ( err = abs(r_real(stat) - r_calc) / abs(r_real(stat)) )
            default: False
            type: bool
        fld : with_err_stat - flag:
                True  - error vs stationary solution will be presented
                False - error vs stationary solution will not be presented
            default: False
            type: bool

        TODO:

        - Finilize.
        '''

        raise NotImplementedError('Is draft.')

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
