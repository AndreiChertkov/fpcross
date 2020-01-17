import time
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib import animation, rc
from copy import deepcopy

import tt

from . import config
from . import Grid
from . import Func
from . import OrdSolver
from .mtr import difscheb
from .utils import tms, PrinterSl


class Solver(object):
    '''
    Class with the fast solver for multidimensional Fokker-Planck equation (FPE)
    d r(x,t) / d t = D Delta( r(x,t) ) - div( f(x,t) r(x,t) ), r(x,0) = r0(x),
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
    6 Call "info" for demonstration (print) of the calculation results.

    Advanced usage:
    - Call "plot" for plot of error and tt-compression factors.
    - Call "plot_tm" for plot of solution and error at selected spatial point.
    - Call "plot_sp" for plot of solution and error at selected time moment.

    PROPS:

    TG - one-dimensional uniform time grid
    type: fpcross.Grid

    SG - one- or multi-dimensional spatial Chebyshev grid
    type: fpcross.Grid

    MD - model for equation with functions and coefficients
    type: fpcross.Model

    FN - function class that keeps the current solution and its interpolation
    type: fpcross.Func

    hst - dictionary with saved (history) values
    type: dict
    fld : T - time moments saved to history
        type: list [opts.n_hst] of float
    fld : R - solutions of the FPE on the spatial grid
        type1: list [0]
        type2: list [opts.n_hst] of np.ndarray [number of points] of float
        type3: list [opts.n_hst] of tt-tensor [number of points] of float
        * Is empty list (type1) if flag opts.with_r_hst is not set.
        * Is type3 if flag with_tt is set and is type2 otherwise.
    fld : rnk_list - tt-ranks of the solution
        type1: list [0]
        type2: list [opts.n_hst] of list [d] of int > 0
        * Is empty list (type1) if flag with_tt is not set.
    fld : rnk_mean - average (effective) tt-rank of the solution
        type1: list [0]
        type2: list [opts.n_hst] of float > 0
        * Is empty list (type1) if flag with_tt is not set.
    fld : cmp_calc - inverse to relative number of points used
                     for drift construction
        type: list [opts.n_hst] of float, > 0
        * This is the calc. economy due to the cross approximation
        * (we calculate only portion 1/cmp_calc of all tensor elements,
        * so this value is full-tensor-size / func-calls, and for the case
        * of "scalar" func-callc it is >= 1).
    fld : cmp_size - inverse to relative number of items for the TT-cores
        type: list [opts.n_hst] of float, > 0
        * This is the compression factor due to the tt-format
        * (we keep in memory only portion 1/cmp_size of all tensor elements,
        * so this value is full-tensor-size / tt_cores_size, and is >= 1
        * for the case of real compression).
    fld : err_real - errors vs analytic (real) solution
        type1: list [0]
        type2: list [opts.n_hst] of float, >= 0
        * Is empty list (type1) if function for real solution is not set.
    fld : err_stat - errors vs stationary solution
        type1: list [0]
        type2: list [opts.n_hst] of float, >= 0
        * Is empty list (type1) if function for stationary solution is not set.
    fld : err_dert - approximated time derivative of the solution (d r / dt)
                     (norm of the derivative devided by the norm of solution)
        type: list [opts.n_hst] of float, >= 0
        * Derivative is calculated as (r_new - r_old) / h.
        * We expects that this value is related to the rhs of the PDF and
        * correlates with the real solution error.
    fld : err_rhsn - norm of the rhs devided by the norm of solution
        type1: list [0]
        type2: list [opts.n_hst] of float, >= 0
        * Is empty list (type1) if flag with_rhs is not set.
        * We expects that this value  correlates with the real solution error.

    res - dictionary with saved values from each time step
          * It has the same fields as hst, but without
          * R, err_real, err_stat and err_rhsn.

    tms - saved durations (in seconds) of the main operations
    type: dict
    fld : init - time spent to init main data structures
        type: float, >= 0
    fld : prep - time spent to prepare required matrices, etc.
        type: float, >= 0
    fld : calc - time spent to perform the calculations
        type: float, >= 0
    fld : calc_init - time to perform operations before the first time step
        type: float, >= 0
    fld : calc_prep - time to perform operations before each time step
        type: float, >= 0
    fld : calc_diff - time to perform calculations for diffusion term
        type: float, >= 0
    fld : calc_conv - time to perform calculations for convection term
        type: float, >= 0
    fld : calc_post - time to perform calculations after each time step
        type: float, >= 0
    fld : calc_last - time to perform operations after the last time step
        type: float, >= 0

    TODO Replace hst.rnk_mean by its computation from hst.rnk_list.
    '''

    def __init__(self, TG, SG, MD, eps=1.E-6, ord=2, with_tt=False):
        '''
        INPUT:

        TG - one-dimensional time grid
        type: fpcross.Grid
        * Only uniform grids are supported in the current version.

        SG - one- or multi-dimensional spatial grid
        type: fpcross.Grid
        * Only Chebyshev "sym" grids (each dimension has the same total
        number of points and limits) are supported in the current version.

        MD - model for equation with functions and coefficients
        type: fpcross.Model

        eps - desired accuracy of the solution
        type: float, >= 1.E-20
        * Is used only for TT-rounding operations and cross approximation
        * in the TT-format (is actual only if with_tt flag is set).
        * For "exact" values (real solution, etc.) while its computation
        * we select 100 times higher accuracy (eps / 100).

        ord - order of the approximation
        type: int
        enum:
            - 1 - for the 1th order splitting and Euler ODE-solver
            - 2 - for the 2th order splitting and Runge-Kutta ODE-solver

        with_tt - flag:
            True  - sparse (tensor train, TT) format will be used
            False - dense (numpy, NP) format will be used
        type: bool

        TODO Add support for non sym Chebyshev grids.
        '''

        if TG.d != 1 or TG.k != 'u':
            raise ValueError('Invalid time grid (should be one-diminsional uniform grid).')

        if SG.k != 'c' or not SG.is_sym():
            raise ValueError('Invalid spatial grid (should be sym Chebyshev grid).')

        if not isinstance(eps, (int, float)) or eps < 1.E-20:
            raise ValueError('Invalid accuracy parameter (should be float >= 1E-20).')

        if not ord in [1, 2]:
            raise ValueError('Invalid order parameter (should be equal to 1 or 2).')

        self.TG = TG
        self.SG = SG
        self.MD = MD
        self.eps = float(eps)
        self.ord = int(ord)
        self.with_tt = bool(with_tt)

        self.FN = Func(self.SG, self.eps, self.with_tt)

        self.opts = {}
        self.init()

    def save(self, fpath):
        '''
        Save solver to file.

        INPUT:

        fpath - absolute or relative path to the file
        type: str

        TODO Add file extension check.

        TODO Save also grids, model etc (add to_dict method to the
             corresponding classes).

        TODO Add try block.
        '''

        data = {
            'opts': self.opts,
            'hst': self.hst,
            'tms': self.tms,
            'tms_list': self.tms_list,
            'res': self.res,
            'res_conv': self.res_conv,
        }
        with open(fpath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, fpath):
        '''
        Load solver from the file.

        INPUT:

        fpath - absolute or relative path to the file
        type: str

        TODO Add file extension check.

        TODO Add try block.

        TODO Add aufo keys extraction (in the loop).
        '''

        with open(fpath, 'rb') as f:
            data = pickle.load(f)

        self.opts = data['opts']
        self.hst = data['hst']
        self.tms = data['tms']
        self.tms_list = data['tms_list']
        self.res = data['res']
        self.res_conv = data['res_conv']

    @tms('init')
    def init(self, opts={}):
        '''
        Init main parameters of the class instance.

        INPUT:

        opts - dictionary with optional parameters
        type: dict
        fld : n_hst - number of points for history
            default: min(10, number_of_time_poins)
            type: int, >= 0, <= number_of_time_poins
            * Errors will be computed and saved for related time moments.
        fld : with_r_hst - flag:
                True  - solution will be saved to history on hist. steps
                False - solution will not be saved to history
            default: False
            type: bool
        fld : with_rhs - flag:
                True  - the rhs of the equation will be calc. on hist. steps
                False - the rhs of the equation will not be calculated
            default: False
            type: bool
        fld : f_prep - function that will be called before each step
                       with self argument (after self.calc_prep)
            default: None
            type: function
        fld : f_post - function that will be called after each step
                       with self argument (after not hst part of self.calc_post)
            default: None
            type: function
        fld : f_post_hst - function that will be called after each hst step
                       with self argument (after self.calc_post and f_post)
            default: None
            type: function
        * Only provided fields will be used. Values of the missed fields will
        * not be changed and will be the same as after the previous call
        * or will be equal to the corresponding default values.

        OUTPUT:

        SL - self
        type: fpcross.Solver

        TODO Check that f_prep, f_post and f_post_hst are functions.
        '''

        def set_opt(name, dflt=None):
            if name in (opts or {}):
                self.opts[name] = opts[name]
            elif not name in self.opts:
                self.opts[name] = dflt

        set_opt('n_hst', np.min([10, self.TG.n0]))
        set_opt('with_r_hst', False)
        set_opt('with_rhs', False)
        set_opt('f_prep', None)
        set_opt('f_post', None)
        set_opt('f_post_hst', None)

        self.opts['n_hst'] = int(self.opts['n_hst']) or 0
        self.opts['with_r_hst'] = bool(self.opts['with_r_hst'])
        self.opts['with_rhs'] = bool(self.opts['with_rhs'])

        self.hst = {
            'T': [],
            'R': [],
            'rnk_list': [],
            'rnk_mean': [],
            'cmp_calc': [],
            'cmp_size': [],
            'err_real': [],
            'err_stat': [],
            'err_dert': [],
            'err_rhsn': [],
        }

        self.tms = {
            'init': 0.,
            'prep': 0.,
            'calc': 0.,
            'calc_init': 0.,
            'calc_prep': 0.,
            'calc_diff': 0.,
            'calc_conv': 0.,
            'calc_post': 0.,
            'calc_last': 0.,
        }

        self.tms_list = {
            'calc_prep': [],
            'calc_diff': [],
            'calc_conv': [],
            'calc_post': [],
        }

        self.res = {
            'T': [],
            'rnk_list': [],
            'rnk_mean': [],
            'cmp_calc': [],
            'cmp_size': [],
            'err_dert': [],
        }

        self.res_conv = []

    @tms('prep')
    def prep(self):
        '''
        Prepare special matrices.

        OUTPUT:

        SL - self
        type: fpcross.Solver

        TODO Check usage of J matrix.

        TODO Add more accurate J matrix construction.
        '''

        self.MD.prep()

        self.D1, self.D2 = difscheb(self.SG, 2)

        h0 = self.TG.h0 if self.ord == 1 else self.TG.h0 / 2.
        Dc = self.MD.D()
        J0 = np.eye(self.SG.n0)
        J0[+0, +0] = 0.
        J0[-1, -1] = 0.
        D0 = self.D2
        self.Z0 = expm(h0 * Dc * J0 @ D0)

        if not self.with_tt:
            if self.opts['with_rhs']:
                self.Xsg = self.SG.comp()
            for i in range(self.SG.d):
                self.Z = self.Z0.copy() if i == 0 else np.kron(self.Z, self.Z0)

    @tms('calc')
    def calc(self, dsbl_print=False):
        '''
        Calculation of the solution.

        INPUT:

        dsbl_print - flag:
            True  - intermediate calculation results will not be printed
            False - intermediate calculation results will be printed
        type: bool

        OUTPUT:

        SL - self
        type: fpcross.Solver
        '''

        self.calc_init(dsbl_print)
        for _ in range(1, self.TG.n0):
            self.calc_prep()
            self.calc_diff()
            self.calc_conv()
            if self.ord == 2:
                self.calc_diff()
            self.calc_post()
        self.calc_last()

    @tms('calc_init')
    def calc_init(self, dsbl_print=False):
        '''
        Some operations before the first computation step.

        INPUT:

        dsbl_print - flag:
            True  - intermediate calculation results will not be printed
            False - intermediate calculation results will be printed
        type: bool

        TODO Check if initial W0 set as r0 is correct.
        '''

        self.m = 0
        self.t = self.TG.l1
        self.PR = PrinterSl(self, with_print=not dsbl_print).init()
        self.FN.init(self.MD.r0).prep()
        self.W0 = self.FN.Y.copy()
        self.FN0 = self.FN.copy()

    @tms('calc_prep', with_list=True)
    def calc_prep(self):
        '''
        Some operations before each computation step.
        '''

        self.m+= 1
        self.t+= self.TG.h0

        if self.opts['f_prep'] is not None:
            self.opts['f_prep'](self)

    @tms('calc_diff', with_list=True)
    def calc_diff(self):
        '''
        One computation step for the diffusion term.

        TODO Check if tt-round is needed at the end.

        TODO Function (if exists) is missed after FN.init.
        '''

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

    @tms('calc_conv', with_list=True)
    def calc_conv(self):
        '''
        One computation step for the convection term.

        TODO Try interpolation of the log.

        TODO Is it requared to add eps to prevent zero division in cross (???) ?

        TODO Try initial guess in the form of the Euler solution?

        TODO Check various numbers of points for ODE-solvers.

        TODO df/dx is not matrix, but vector in the model. Add more correct
             name for the corresponding function.
        '''

        def func(y, t):
            '''
            Compute rhs for the system of drift equations
            d x / dt = f(x, t)
            d w / dt = - Tr[ d f(x, t) / d x ] w.

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
                -np.sum(f1, axis=0) * r
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

            TG = Grid(d=1, n=2, l=[self.t - self.TG.h0, self.t], k='u')
            kd = 'eul' if self.ord == 1 else 'rk4'
            X0 = OrdSolver(TG, kind=kd, is_rev=True).init(self.MD.f0).comp(X)
            w0 = FN.comp(X0)
            y0 = np.vstack([X0, w0])
            y1 = OrdSolver(TG, kind=kd).init(func).comp(y0)
            w1 = y1[-1, :]

            # if self.with_tt and np.linalg.norm(w1) < 1.E-20:
            #     w1+= 1.E-20 # To prevent zero division in the cross appr.

            return w1

        # self.FN.calc()
        # mult = 1. / self.FN.comp_int()
        # self.FN.Y = mult * self.FN.Y
        # self.FN.A = mult * self.FN.A
        FN = self.FN.copy()
        FN.calc()

        self.FN.init(step, opts={
            'nswp': 200,
            'kickrank': 1,
            'rf': 2,
            'Y0': self.W0,
            #'with_Y_hst': True,
        }).prep()

        self.res_conv.append(deepcopy(self.FN.res))

        self.W0 = self.FN.Y.copy()

    @tms('calc_post', with_list=True)
    def calc_post(self):
        '''
        Check result of the current computation step.

        TODO Remove self.FN.calc() (and change algorithm for comp_int) (???).

        TODO Add initial guess r to rt and maybe rs.
        '''

        t_hst = int(self.TG.n0/self.opts['n_hst']) if self.opts['n_hst'] else 0.
        is_hst = t_hst and (self.m % t_hst == 0 or self.m == self.TG.n0 - 1)

        if self.opts['f_post'] is not None:
            self.opts['f_post'](self)

        self.FN.calc()                   # REMOVE ???
        mult = 1. / self.FN.comp_int()
        self.FN.Y = mult * self.FN.Y
        self.FN.A = mult * self.FN.A

        self.exam(self.res)
        if is_hst:
            self.exam(self.hst, is_hst=True)

        self.FN0 = self.FN.copy()

        if not is_hst:
            self.PR.refr()
            return

        msg = '| At T=%-6.1e : '%self.t + ' ' * 100

        if True:
            msg+= '  Edert=%-6.1e'%self.hst['err_dert'][-1]
        if self.opts['with_rhs']:
            msg+= '  Erhsn=%-6.1e'%self.hst['err_rhsn'][-1]
        if self.MD.with_rt():
            msg+= '  Ereal=%-6.1e'%self.hst['err_real'][-1]
        if self.MD.with_rs():
            msg+= '  Estat=%-6.1e'%self.hst['err_stat'][-1]
        if self.with_tt:
            msg+= ' r=%-6.2e'%self.hst['rnk_mean'][-1]

        self.PR.refr(msg)
        self.FN0 = self.FN.copy()

        if self.opts['f_post_hst'] is not None:
            self.opts['f_post_hst'](self)

    @tms('calc_last')
    def calc_last(self):
        '''
        Some operations after the final computation step.
        '''

        self.FN.calc()
        mult = 1. / self.FN.comp_int()
        self.FN.Y = mult * self.FN.Y
        self.FN.A = mult * self.FN.A

        self.PR.close()

    def comp(self, X):
        '''
        Compute calculated solution at given spatial points.

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points]

        OUTPUT:

        r - calculated solution in the given points
        type: ndarray [number of points] of float

        TODO Add support for one-point input.
        '''

        if self.FN.A is None:
            raise ValueError('Solution of the equation is not calculated yet. Call "prep" and "calc" functions before.')

        return self.FN.comp(X)

    def comp_rhs(self, t=None, is_real=False, is_stat=False):
        '''
        Compute the norm of the right hand side (rhs) of the FPE
        divided by norm of the solution.

        INPUT:

        t - time for computation
        type1: None
        type2: float
        * If is not set (type1) then the current solver time will be used.

        is_real - flag:
            True  - will be calculated for real (analytic) solution
            False - will not be calculated for real (analytic) solution

        is_stat - flag:
            True  - will be calculated for stationary solution
            False - will not be calculated for stationary solution
        type: bool

        OUTPUT:

        res - norm of rhs divided by norm of the solution
        type: float, >= 0

        TODO Check.
        '''

        if is_real and is_stat:
            raise ValueError('Only one option (real or stat) may be selected.')

        t = self.t if not t else t
        res = None

        FN_r = self.FN.copy()
        if is_real:
            FN_r.init(self.MD.rt).prep()
        if is_stat:
            FN_r.init(self.MD.rs).prep()

        if self.with_tt:
            D = self.MD.D()
            r = FN_r.Y

            for k in range(self.SG.d):
                G = tt.tensor.to_list(r)
                G[k] = np.einsum('ij,kjm->kim', self.D2, G[k])
                v = tt.tensor.from_list(G) * D
                v = v.round(self.eps)
                res = v.copy() if res is None else (res + v).round(self.eps)

                def func(X):
                    return self.MD.f0(X, t)[k, :]

                f = self.FN.copy(is_init=True).init(func).prep().Y
                G = tt.tensor.to_list((r * f).round(self.eps))
                G[k] = np.einsum('ij,kjm->kim', self.D1, G[k])
                v = tt.tensor.from_list(G)
                v = v.round(self.eps)
                res = v.copy() if res is None else (res - v).round(self.eps)

            res = res.norm() / r.norm()

        else:
            X = self.Xsg
            f = self.MD.f0(X, t)
            D = self.MD.D()
            r = FN_r.Y.reshape(-1, order='F')

            res = np.zeros(r.shape[0])
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

                res+= _D2 @ (r * D) - _D1 @ (r * f[k, :])

            res = np.linalg.norm(res) / np.linalg.norm(r)

        return res

    def exam(self, obj=None, eps_mult=100, is_hst=False):
        def _err_calc(r_calc, r_real):
            dr = r_real - r_calc
            n0 = r_real.norm() if self.with_tt else np.linalg.norm(r_real)
            dn = dr.norm() if self.with_tt else np.linalg.norm(dr)
            return dn / n0 if n0 > 0 else dn

        res = {}

        if True:
            res['T'] = self.t

        if is_hst and self.opts['with_r_hst']:
            res['R'] = self.FN.Y.copy()

        if self.with_tt:
            res['rnk_list'] = self.FN.Y.r.copy()
            res['rnk_mean'] = self.FN.Y.erank

            res['cmp_calc'] = self.res_conv['evals'] # ZZZ
            for i in range(self.SG.d):
                res['cmp_calc']/= self.SG.n[i]
            res['cmp_calc'] = 1. / res['cmp_calc']

            res['cmp_size'] = 0.
            for i in range(self.SG.d):
                res['cmp_size']+= self.FN.Y.r[i]*self.SG.n[i]*self.FN.Y.r[i+1]
            for i in range(self.SG.d):
                res['cmp_size']/= self.SG.n[i]
            res['cmp_size'] = 1. / res['cmp_size']

        if is_hst and (self.MD.with_rt() or self.MD.with_rs()):
            FN = self.FN.copy(eps=self.FN.eps/eps_mult, is_init=True)

        if True:
            res['err_dert'] = _err_calc(self.FN0.Y, self.FN.Y) / self.TG.h0

        if is_hst and self.opts['with_rhs']:
            res['err_rhsn'] = self.comp_rhs()

        if is_hst and self.MD.with_rt():
            FN.init(lambda X: self.MD.rt(X, self.t)).prep()
            res['err_real'] = _err_calc(self.FN.Y, FN.Y)

        if is_hst and self.MD.with_rs():
            FN.init(lambda X: self.MD.rs(X)).prep()
            res['err_stat'] = _err_calc(self.FN.Y, FN.Y)

        if obj is not None:
            for name in res.keys():
                obj[name].append(res[name])

        return res

    def info(self, is_ret=False):
        '''
        Present information about the last computation.

        INPUT:

        is_ret - flag:
            True  - return string info
            False - print string info
        type: bool

        OUTPUT:

        s - (if is_ret) string with info
        type: str
        '''

        opts, hst, tms = self.opts, self.hst, self.tms
        n_hst = opts['n_hst']

        s = '------------------ Solver\n'
        s+= 'Format    : %1dD, '%self.SG.d
        s+= 'TT, eps= %8.2e '%self.eps if self.with_tt else 'NP '
        s+= '[order=%d]\n'%self.ord

        s+= 'Hst pois  : %s \n'%('%d'%n_hst if n_hst else 'None')
        s+= 'Hst with r: %s \n'%('Yes' if opts['with_r_hst'] else 'No')

        if len(hst['err_dert']):
            s+= 'd r / d t : %8.2e\n'%hst['err_dert'][-1]
        if len(hst['err_rhsn']):
            s+= 'Err  rhs  : %8.2e\n'%hst['err_rhsn'][-1]
        if len(hst['err_real']):
            s+= 'Err  real : %8.2e\n'%hst['err_real'][-1]
        if len(hst['err_stat']):
            s+= 'Err  stat : %8.2e\n'%hst['err_stat'][-1]

        s+= 'Time full : %8.2e \n'%(tms['prep'] + tms['calc'])
        s+= 'Time prep : %8.2e \n'%tms['prep']
        s+= 'Time calc : %8.2e \n'%tms['calc']
        s+= '    .init : %8.2e \n'%tms['calc_init']
        s+= '    .prep : %8.2e \n'%tms['calc_prep']
        s+= '    .diff : %8.2e \n'%tms['calc_diff']
        s+= '    .conv : %8.2e \n'%tms['calc_conv']
        s+= '    .post : %8.2e \n'%tms['calc_post']
        s+= '    .last : %8.2e \n'%tms['calc_last']

        if is_ret:
            return s + '\n'
        print(s)

    def anim(self, ffmpeg_path, delt=50):
        '''
        Build animation for solution on the spatial grid vs time.

        INPUT:

        ffmpeg_path - path to the ffmpeg executable
        type: str

        delt - number of frames per second
        type: int, > 0

        TODO Finilize.
        '''

        raise NotImplementedError('Is draft.')

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

    def plot(self):
        '''
        Plot results of the last computation (errors, times, TT-ranks
        and compression factors).
        '''

        kind = ' (N=%d^%d, M=%d points)'%(self.SG.n0, self.SG.d, self.TG.n0)
        conf = config['opts']['plot']

        Tall = self.res['T']
        Thst = self.hst['T']

        err_dert = self.res['err_dert']
        err_rhsn = self.hst['err_rhsn']
        err_stat = self.hst['err_stat']
        err_real = self.hst['err_real']

        tms_prep = self.tms_list['calc_prep']
        tms_diff = self.tms_list['calc_diff']
        tms_conv = self.tms_list['calc_conv']
        tms_post = self.tms_list['calc_post']

        rnk_mean = self.hst['rnk_mean']
        cmp_calc = self.hst['cmp_calc']
        cmp_size = self.hst['cmp_size']

        if self.ord != 1:
            tms_diff = tms_diff[0::2]

        opts = conf['fig']['base_2_2' if self.with_tt else 'base_1_2']
        fig = plt.figure(**opts)

        opts = conf['grid']['base_2_2' if self.with_tt else 'base_1_2']
        gs = mpl.gridspec.GridSpec(**opts)

        if True:
            ax = fig.add_subplot(gs[0, 0])
            ax.set_title('Relative error' + kind)
            ax.set_xlabel('Time')
            if len(err_dert):
                ax.plot(Tall, err_dert, **{
                    **conf['line']['l4'],
                    'linestyle': '--',
                    'linewidth': 4,
                    'label': 'numerical derivative',
                })
            if len(err_rhsn):
                ax.plot(Thst, err_rhsn, **{
                    **conf['line']['l11'],
                    'label': 'relative rhs',
                })
            if len(err_stat):
                ax.plot(Thst, err_stat, **{
                    **conf['line']['l7'],
                    'label': 'vs analytic stat. soltion',
                })
            if len(err_real):
                ax.plot(Thst, err_real, **{
                    **conf['line']['l6'],
                    'label': 'vs analytic solution',
                })
            ax.legend(loc='best')
            ax.semilogy()

        if True:
            ax = fig.add_subplot(gs[0, 1])
            ax.set_title('Step duration'+ kind)
            ax.set_xlabel('Time')
            opts = { 'color': 'orange', 'marker': '*', 'markersize': 8, 'markeredgecolor': 'orange' }
            ax.plot(Tall, tms_prep, **{
                **conf['line']['l1'],
                'label': 'Prepare before step',
            })
            ax.plot(Tall, tms_diff, **{
                **conf['line']['l2'],
                'label': 'Diffusion part',
            })
            ax.plot(Tall, tms_conv, **{
                **conf['line']['l4'],
                'label': 'Convection part',
            })
            ax.plot(Tall, tms_post, **{
                **conf['line']['l5'],
                'label': 'Check after step',
            })
            ax.legend(loc='best')
            ax.semilogy()

        if self.with_tt:
            ax = fig.add_subplot(gs[1, 0])
            ax.set_title('TT-erank'+ kind)
            ax.set_xlabel('Time')
            opts = { 'color': 'orange', 'marker': '*', 'markersize': 8, 'markeredgecolor': 'orange' }
            ax.plot(Thst, rnk_mean, **{
                **conf['line']['l14'],
            })

        if self.with_tt:
            ax = fig.add_subplot(gs[1, 1])
            ax.set_title('TT-compression factor'+ kind)
            ax.set_xlabel('Time')
            opts = { 'color': 'orange', 'marker': '*', 'markersize': 8, 'markeredgecolor': 'orange' }
            ax.plot(Thst, cmp_calc, **{
                **conf['line']['l13'],
                'label': 'Function evaluations',
            })
            ax.plot(Thst, cmp_size, **{
                **conf['line']['l15'],
                'label': 'Memory usage',
            })
            ax.legend(loc='best')
            ax.semilogy()

        plt.show()

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

        TODO Finilize.
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

        TODO Finilize.
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
