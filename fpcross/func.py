import os
import sys
import time
import numpy as np

import tt
from tt.cross.rectcross import cross

from . import polycheb

class Func(object):
    '''
    Class for the fast multidimensional function interpolation
    by Chebyshev polynomials in the dense (numpy) or sparse
    (tensor train (TT) with cross approximation) format
    using Fast Fourier Transform (FFT).

    Basic usage:
    1 Initialize class instance with grid, accuracy and format parameters.
    2 Call "init" with function of interest and interpolation options.
    3 Call "prep" for construction of the function on the spatial grid.
    4 Call "calc" for interpolation process.
    5 Call "comp" for approximation of function values on any given points.
    6 Call "info" for demonstration of calculation results.
    7 Call "copy" to obtain new instance with the same parameters and results.

    Advanced usage:
    - Call "test" to obtain interpolation accuracy for set of non grid points.

    PROPS:

    Y - tensor of function values on nodes of the Chebyshev grid
    type: ndarray or TT-tensor [*(numbers of points)] of float
    * Is set in self.init or calculated in self.prep.

    A - tensor of interpolation coefficients
    type: ndarray or TT-tensor [*(numbers of points)] of float
    * Is calculated in self.calc.

    tms - saved durations of the main operations
    type: dict
    fld : prep - construction of the function values on the Chebyshev grid
        type: float, >= 0
    fld : calc - construction of interpolation coefficients
        type: float, >= 0
    fld : comp - computation of the interpolant in one given spatial point
        type: float, >= 0
    fld : func - computation of the base function in one given spatial point
        type: float, >= 0

    res - results of the computations (tt-ranks, etc.)
    type: dict

    err - abs. value of the relative error of result for some non grid points
    type: np.ndarray [number of test points] of float, >= 0.
    * Is calculated in self.test.
    '''

    def __init__(self, SG, eps=1.E-6, with_tt=False):
        '''
        INPUT:

        SG - one- or multi-dimensional spatial grid for function interpolation
        type: fpcross.Grid
        * Only Chebyshev grids are supported in the current version.

        eps - is the desired accuracy of the approximation
        type: float, >= 1.E-20
        * Is used in tt-round and tt-cross approximation operations.

        with_tt - flag:
            True  - sparse (tensor train, TT) format will be used
            False - dense (numpy, NP) format will be used
        type: bool
        '''

        self.SG = SG
        self.eps = float(eps)
        self.with_tt = bool(with_tt)

        if self.SG.kind != 'c':
            raise ValueError('Invalid spatial grid (should be Chebyshev).')

        if self.eps < 1.E-20:
            raise ValueError('Invalid accuracy parameter (should be > 1.E-20).')

        self.opts = {}
        self.init()

    def copy(self, opts=None, is_init=False):
        '''
        Create a copy of the class instance.

        INPUT:

        opts - dictionary with optional parameters
        type1: None
        type2: dict (see self.init for details)
        * If is set (type2), then init method with this dict will be called.

        is_init - flag:
            True  - interpolation result will not be copied
            False - interpolation result (if exists) will be also copied
        type: bool

        OUTPUT:

        FN - new class instance
        type: fpcross.Func

        TODO:

        - Add support for new parameters for __init__.
        '''

        FN = Func(self.SG.copy(), self.eps, self.with_tt)
        FN.init(opts=self.opts).init(opts=opts)

        if is_init: return FN

        FN.f = self.f
        FN.Y = self.Y.copy() if self.Y is not None else None
        FN.A = self.A.copy() if self.A is not None else None

        FN.tms = self.tms.copy()
        FN.res = self.res.copy()
        FN.err = self.err.copy() if self.err is not None else None

        return FN

    def init(self, f=None, Y=None, opts={}):
        '''
        Init the main parameters of the class instance.

        INPUT:

        f - function that calculate tensor elements for given spatial points
        type1: None
        type2: function
            inp:
                X - points for calculation
                type: ndarray [dimensions, number of points] of float

                I - indices of X points (if opts.is_f_with_i flag is set)
                type: ndarray [dimensions, number of points] of int
            out:
                Y - function values on X points
                type: ndarray [number of points] of float

        Y - tensor of function values on nodes of the Chebyshev grid
        type1: None
        type2: ndarray [*(numbers of points)] of float
        type3: TT-tensor [*(numbers of points)] of float
        * Should be in the TT-format (type3) if self.with_tt flag is set.
        * If is set (type2 or type3), then function f will not be used for
        * train data collection, but may be used in self.test for check.

        opts - dictionary with optional parameters
        type: dict
        fld : is_f_with_i - flag:
                True  - while function call the second argument with points
                        indices will be provided
                False - only points will be provided
            default: False
            type: bool
        fld : nswp - for cross approximation
            default: 200
            type: int, > 0
        fld : kickrank - for cross approximation
            default: 1
            type: int, > 0
        fld : rf - for cross approximation
            default: 2
            type: int, > 0
        fld : Y0 - initial guess for cross approximation
            default: None (random TT-tensor will be used)
            type1: None
            type2: ndarray [*n] of float
            type3: TT-tensor [*n] of float
        * Only provided fields will be used. Values of the missed fields will
        * not be changed and will be the same as after the previous call
        * or will be equal to the corresponding default values.

        OUTPUT:

        FN - self
        type: fpcross.Func

        TODO:

        - Add support for functions that can calculate only one point
          by one function call and for other function inp/out formats.
        '''

        self.f = f
        self.Y = Y
        self.A = None

        self.tms = {
            'prep': 0., 'calc': 0., 'comp': 0., 'func': 0.
        }
        self.res = {
            'evals': 0, 't_func': 0., 'iters': 0,
            'err_rel': 0., 'err_abs': 0., 'erank': 0.
        }
        self.err = None

        def set_opt(name, dflt=None):
            if name in (opts or {}): self.opts[name] = opts[name]
            elif not name in self.opts: self.opts[name] = dflt

        set_opt('is_f_with_i', False)
        set_opt('nswp', 200)
        set_opt('kickrank', 1)
        set_opt('rf', 2)
        set_opt('Y0', None)

        return self

    def prep(self):
        '''
        Construct function values on the spatial grid.

        OUTPUT:

        FN - self
        type: fpcross.Func

        TODO:

        - If Y0 is not set, how select best random value (rank, etc.)?

        - Set more accurate algorithm for tt-round of initial guess.
        '''

        if self.Y is not None:
            raise ValueError('Function values are already prepared.')

        if self.f is None:
            raise ValueError('Function is not set. Can not prepare.')

        self.tms['prep'] = time.perf_counter()

        if self.with_tt: # TT-format
            log_file = './__tt-cross_tmp.txt'

            def func(ind):
                ind = ind.astype(int)
                t = time.perf_counter()
                X = self.SG.comp(ind.T)
                Y = self.f(X, ind.T) if self.opts['is_f_with_i'] else self.f(X)
                self.tms['func']+= time.perf_counter() - t
                self.res['evals']+= X.shape[1]
                return Y

            if self.opts['Y0'] is None:
                Z = tt.rand(self.SG.n, self.SG.d, 1)
            else:
                Z = self.opts['Y0'].copy()

                rmax = None
                for n, r in zip(self.SG.n, Z.r[1:]):
                    if r > n and (rmax is None or rmax > n): rmax = n
                if rmax is not None: Z = Z.round(rmax=rmax)

            try:
                log = open(log_file, 'w')
                stdout0 = sys.stdout
                sys.stdout = log

                self.Y = cross(func, Z, eps=self.eps,
                    nswp=self.opts['nswp'], kickrank=self.opts['kickrank'],
                    rf=self.opts['rf'], verbose=True
                )
            finally:
                log.close()
                sys.stdout = stdout0

            self.tms['func']/= (self.res['evals'] or 1)

            log = open(log_file, 'r')
            res = log.readlines()[-1].split('swp: ')[1]
            self.res['iters'] = int(res.split('/')[0])+1
            res = res.split('er_rel = ')[1]
            self.res['err_rel'] = float(res.split('er_abs = ')[0])
            res = res.split('er_abs = ')[1]
            self.res['err_abs'] = float(res.split('erank = ')[0])
            res = res.split('erank = ')[1]
            self.res['erank'] = float(res.split('fun_eval')[0])
            log.close()
            os.remove(log_file)

        else:            # NP-format
            self.tms['func'] = time.perf_counter()

            X = self.SG.comp()
            if self.opts['is_f_with_i']:
                I = self.SG.comp(is_ind=True)
                self.Y = self.f(X, I)
            else:
                self.Y = self.f(X)
            self.Y = self.Y.reshape(self.SG.n, order='F')

            self.tms['func'] = (time.perf_counter()-self.tms['func'])/X.shape[1]

        self.tms['prep'] = time.perf_counter() - self.tms['prep']

        return self

    def calc(self):
        '''
        Build tensor of interpolation coefficients according to training data.

        OUTPUT:

        FN - self
        type: fpcross.Func

        TODO:

        - Should we round A tensor after construction?
        '''

        if self.Y is None:
            raise ValueError('Train data is not set. Can not build interpolation coefficients. Call "prep" before.')

        self.tms['calc'] = time.perf_counter()

        if self.with_tt: # TT-format
            G = tt.tensor.to_list(self.Y)

            for i in range(self.SG.d):
                G_sh = G[i].shape
                G[i] = np.swapaxes(G[i], 0, 1)
                G[i] = G[i].reshape((G_sh[1], -1))
                G[i] = Func.interpolate_cheb(G[i])
                G[i] = G[i].reshape((G_sh[1], G_sh[0], G_sh[2]))
                G[i] = np.swapaxes(G[i], 0, 1)

            self.A = tt.tensor.from_list(G)
            self.A = self.A.round(self.eps)

        else:            # NP-format
            self.A = self.Y.copy()

            for i in range(self.SG.d):
                n_ = self.SG.n.copy()
                n_[[0, i]] = n_[[i, 0]]
                self.A = np.swapaxes(self.A, 0, i)
                self.A = self.A.reshape((self.SG.n[i], -1), order='F')
                self.A = Func.interpolate_cheb(self.A)
                self.A = self.A.reshape(n_, order='F')
                self.A = np.swapaxes(self.A, 0, i)

        self.tms['calc'] = time.perf_counter() - self.tms['calc']

        return self

    def comp(self, X, z=0.):
        '''
        Compute values of interpolated function in given X points.

        INPUT:

        X - values of the spatial variable
        type1: list [dimensions, number of points] of float
        type2: ndarray [dimensions, number of points] of float

        z - value for points outside the interpolation limits
        type: float
        * If some points are not belong to the interpolation limits, then
        * the corresponding values will be set to the z-value.

        OUTPUT:

        Y - approximated values of the function in given points
        type: ndarray [number of points] of float

        TODO:

        - Vectorize calculations for points vector if possible.

        - Add support for 1D input.

        - Check if correct calculate Cheb. pol. until max(n) for all dimensions.
        '''

        if self.A is None:
            raise ValueError('Interpolation is not done. Can not compute values of the function. Call "calc" before.')

        if not isinstance(X, np.ndarray): X = np.array(X)

        self.tms['comp'] = time.time()

        Y = np.ones(X.shape[1]) * z
        T = polycheb(X, np.max(self.SG.n), self.SG.l)

        if self.with_tt: # TT-format
            G = tt.tensor.to_list(self.A)

            for j in range(X.shape[1]):
                if self.SG.is_out(X[:, j]): continue

                i = 0
                Q = np.einsum('riq,i->rq', G[i], T[:self.SG.n[i], i, j])
                for i in range(1, self.SG.d):
                    Q = Q @ np.einsum('riq,i->rq', G[i], T[:self.SG.n[i], i, j])

                Y[j] = Q[0, 0]

        else:            # NP-format
            for j in range(X.shape[1]):
                if self.SG.is_out(X[:, j]): continue

                Q = self.A.copy()
                for i in range(self.SG.d):
                    Q = np.tensordot(Q, T[:Q.shape[0], i, j], axes=([0], [0]))

                Y[j] = Q

        self.tms['comp'] = (time.time() - self.tms['comp']) / X.shape[1]

        return Y

    def comp_int(self):
        '''
        Compute integral of the function on the full domain.

        OUTPUT:

        v - value of the integral
        type: float
        '''

        if self.A is None:
            raise ValueError('Interpolation is not done. Can not compute integral of the function. Call "calc" before.')

        if not self.SG.is_sym():
            raise ValueError('Can integrate only for symmetric spatial grid.')

        if self.with_tt: # TT-format
            G = tt.tensor.to_list(self.A)
            v = np.array([[1.]])

            for k in range(self.SG.d):
                G_sh = G[k].shape                       # ( R_{k-1}, N_k, R_k )
                G[k] = np.swapaxes(G[k], 0, 1)
                G[k] = G[k].reshape(G_sh[1], -1)
                G[k] = Func.integrate_cheb(G[k])
                G[k] = G[k].reshape(G_sh[0], G_sh[2])

                v = v @ G[k]
                v*= (self.SG.l[k, 1] - self.SG.l[k, 0]) / 2.

            v = v[0, 0]

        else:            # NP-format
            v = self.A.copy()

            for k in range(self.SG.d):
                v = v.reshape(self.SG.n[k], -1)
                v = Func.integrate_cheb(v)
                v*= (self.SG.l[k, 1] - self.SG.l[k, 0]) / 2.

            v = v[0]

        return v

    def info(self, n_test=None, is_test_u=False, is_ret=False):
        '''
        Present info about interpolation result, including error check.

        INPUT:

        n_test - number of points for error check
        type1: None
        type2: int, >= 0
        * If is set (is not None and is greater than zero) and interpolation is
        * ready, then interpolation result will be checked on a set
        * of random points from the uniform distribution or on a set of
        * uniform points with proper limits (depends on the value of is_test_u).

        is_test_u - flag:
            True  - uniform points will be used for error check
            False - random points will be used for error check
        type: bool
        * Note that if this flag is set, then less than n_test points will be
        * used for the test due to rounding of the d-root and using of the only
        * inner grid points.

        is_ret - flag:
            True  - return string info
            False - print string info
        type: bool

        OUTPUT:

        s - (if is_out) string with info
        type: str
        '''

        if self.A is not None and self.f is not None and n_test:
            self.test(n_test, is_test_u)

        s = '------------------ Function  \n'
        s+= 'Format           : %1dD, '%self.SG.d
        s+= 'TT, eps= %8.2e\n'%self.eps if self.with_tt else 'NP\n'

        if True:
            s+= '--> Time (sec.)  |       \n'
            s+= 'Prep             : %8.2e \n'%self.tms['prep']
            s+= 'Calc             : %8.2e \n'%self.tms['calc']
            s+= 'Comp (average)   : %8.2e \n'%self.tms['comp']
            s+= 'Func (average)   : %8.2e \n'%self.tms['func']

        if self.A is not None and self.f is not None and n_test:
            s+= '--> Test         |       \n'
            s+= 'Random points    : %8s   \n'%('No' if is_test_u else 'Yes')
            s+= 'Number of points : %8d   \n'%self.err.size
            s+= 'Error (max)      : %8.2e \n'%np.max(self.err)
            s+= 'Error (mean)     : %8.2e \n'%np.mean(self.err)
            s+= 'Error (min)      : %8.2e \n'%np.min(self.err)

        if self.with_tt:
            with_ig = 'No' if self.opts['Y0'] is None else 'Yes'

            s+= '--> Cross params |       \n'
            s+= 'Initial guess    : %8s   \n'%with_ig
            s+= 'nswp             : %8d   \n'%self.opts['nswp']
            s+= 'kickrank         : %8d   \n'%self.opts['kickrank']
            s+= 'rf               : %8.2e \n'%self.opts['rf']

        if self.with_tt:
            s+= '--> Cross result | \n'
            s+= 'Func. evaluations: %8d   \n'%self.res['evals']
            s+= 'Cross iterations : %8d   \n'%self.res['iters']
            s+= 'Av. tt-rank      : %8.2e \n'%self.res['erank']
            s+= 'Cross err (rel)  : %8.2e \n'%self.res['err_rel']
            s+= 'Cross err (abs)  : %8.2e \n'%self.res['err_abs']

        if not s.endswith('\n'): s+= '\n'
        if is_ret: return s
        print(s[:-1])

    def test(self, n=100, is_u=False):
        '''
        Calculate interpolation error on a set of random (from the uniform
        distribution) or uniform points with proper limits.

        INPUT:

        n - number of points for error check
        type: int, > 0

        is_u - flag:
            True  - uniform points will be used
            False - random points will be used
        type: bool

        OUTPUT:

        err - absolute value of relative error for the generated random points
        type: np.ndarray [number of points] of float
        * Note that if is_u flag is set, then less than n points will be used
        * for the test due to rounding of the d-root and using of the only
        * inner grid points, hence number of points <= n.

        TODO:

        - Add support for absolute error.
        '''

        if self.f is None:
            s = 'Function for interpolation is not set. Can not test.'
            raise ValueError(s)

        if self.opts['is_f_with_i']:
            s = 'Function for interpolation requires indices. Can not test.'
            raise ValueError(s)

        if is_u:
            X = self.SG.copy(n=n**(1./self.SG.d), kind='u').comp(is_inner=True)
        else:
            X = self.SG.rand(n)

        u = self.f(X)
        v = self.comp(X)
        self.err = np.abs((u - v) / u)

        return self.err

    @staticmethod
    def interpolate_cheb(Y):
        '''
        Find coefficients a_i for interpolation of 1D functions by Chebyshev
        polynomials f(x) = \sum_{i} (a_i * T_i(x)) using Fast Fourier Transform.

        It can find coefficients for several functions on the one call,
        if all functions have equal numbers of grid points.

        INPUT:

        Y - values of function at the nodes of the Chebyshev grid
            x_j = cos(\pi j / (N - 1)), j = 0, 1 ,..., N-1,
            where N is a number of points
        type1: list [number of points] of float
        type2: ndarray [number of points] of float
        type3: list [number of points, number of functions] of float
        type4: ndarray [number of points, number of functions] of float
        * It may be of type1 or type2 only in the case of only one function.

        OUTPUT:

        A - constructed matrix of coefficients
        type: ndarray [number of points, number of functions] of float
        '''

        if not isinstance(Y, np.ndarray): Y = np.array(Y)
        if len(Y.shape) == 1: Y = Y.reshape(-1, 1)
        if len(Y.shape) != 2:
            raise ValueError('Invalid shape for function values.')

        n = Y.shape[0]
        V = np.vstack([Y, Y[n-2 : 0 : -1, :]])
        A = np.fft.fft(V, axis=0).real
        A = A[:n, :] / (n - 1)
        A[0, :] /= 2.
        A[n-1, :] /= 2.

        return A

    @staticmethod
    def integrate_cheb(A):
        '''
        Integrate one-dimensional function f(x) = \sum_{i} (A_i * T_i(x)),
        using known coefficients for interpolation by Chebyshev polynomials,
        on the interval (-1, 1).

        It can find coefficients for several functions on the one call.

        INPUT:

        A - interpolation coefficients
        type1: list [number of points] of float
        type2: ndarray [number of points] of float
        type3: list [number of points, number of functions] of float
        type4: ndarray [number of points, number of functions] of float
        * It may be of type1 or type2 only in the case of only one function.

        OUTPUT:

        v - value of the integral
        type: float
        '''

        if not isinstance(A, np.ndarray): A = np.array(A)
        if len(A.shape) == 1: A = A.reshape(-1, 1)

        n = np.arange(A.shape[0])[::2]
        n = np.repeat(n.reshape(-1, 1), A.shape[1], axis=1)

        return np.sum(A[::2, :] * 2. / (1. - n**2), axis=0)
