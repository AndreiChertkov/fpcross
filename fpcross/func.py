import sys
import time
import numpy as np

import tt
from tt.cross.rectcross import cross

from . import polynomials_cheb

class Func(object):
    '''
    Class for the fast multidimensional function interpolation
    by Chebyshev polynomials in the dense (numpy) or sparse
    (tensor train (TT) with cross approximation) format
    using Fast Fourier Transform (FFT).

    Basic usage:
    1 Initialize class instance with grid, format and accuracy parameters.
    2 Call "init" with function of interest and interpolation options.
    3 Call "prep" for construction of the function on the spatial grid.
    4 Call "calc" for interpolation process.
    5 Call "comp" for approximation of function values on any given points.
    6 Call "info" for demonstration of calculation results.
    7 Call "copy" to obtain new instance with the same parameters and results.

    Advanced usage:
    - Call "test" to obtain interpolation accuracy for set of random points.

    PROPS:

    Y

    A
    '''

    def __init__(self, SG, eps=1.E-6, with_tt=True):
        '''
        INPUT:

        SG - one- or multi-dimensional spatial grid for function interpolation
        type: fpcross.Grid
        * Only Chebyshev grids are supported in the current version.

        eps - is the desired accuracy of the approximation
        type: float, > 0
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

    def init(self, f=None, Y=None, opts={}):
        '''
        Init the main parameters of the class instance.

        INPUT:

        f - function that calculate tensor elements for given spatial points
        type: function
            inp:
                X - points for calculation
                type: ndarray [dimensions, number of points] of float

                I - indices of X points (if opts.is_f_with_i flag is set)
                type: ndarray [dimensions, number of points] of int
            out:
                Y - function values on X points
                type: ndarray [number of points] of float

        Y - tensor of function values on nodes of the Chebyshev grid
        type1: ndarray [*(numbers of points)] of float
        type2: TT-tensor [*(numbers of points)] of float
        * Should be in the TT-format (type2) if self.with_tt flag is set.
        * If is set, then function f will not be used for train data collection,
        * but may be used in self.test for check of the interpolation result.

        opts - dictionary with optional parameters
        type: dict
        fld: is_f_with_i - flag:
                True  - while function call the second argument with points
                        indices will be provided
                False - only points will be provided
            default: False
            type: bool
        fld:nswp - for cross approximation
            default: 200
            type: int, > 0
        fld:kickrank - for cross approximation
            default: 1
            type: int, > 0
        fld:rf - for cross approximation
            default: 2
            type: int, > 0
        fld:Y0 - initial guess for cross approximation
            default: None (random TT-tensor will be used)
            type: ndarray or TT-tensor [*n] of float
        * Only provided fields will be used. Values of the missed fields will
        * not be changed and will be the same as after the previous call.

        TODO: Check case for update with Y0 opt changed to None.
        '''

        self.f = f
        self.Y = Y
        self.A = None

        self.tms = {      # Saved times (durations)
            'prep': 0.,   # training data collection
            'calc': 0.,   # interpolation coefficients calculation
            'comp': 0.,   # interpolation computation (average time for 1 poi)
            'func': 0.,   # function computation (average time for 1 poi)
        }
        self.res = None
        self.err = None

        def set_opt(name, dflt=None):
            v = (opts or {}).get(name)
            if v is not None:
                self.opts[name] = v
                return

            v = self.opts.get(name)
            if v is not None:
                return

            self.opts[name] = dflt

        set_opt('is_f_with_i', False)
        set_opt('nswp', 200)
        set_opt('kickrank', 1)
        set_opt('rf', 2)
        set_opt('Y0', None)

    def copy(self, is_full=True):
        '''
        Create a copy of the class instance.

        INPUT:

        is_full - flag:
            True  - interpolation result (if exists) will be also copied
            False - only parameters will be copied
        type: bool

        OUTPUT:

        FN - new class instance
        type: fpcross.Func

        TODO: Add argument with changed opts (eps etc) ?
        '''

        FN = Func(self.SG.copy(), self.eps, self.with_tt)

        if not is_full: return FN

        FN.f = self.f
        FN.Y = self.Y.copy() if self.Y is not None else None
        FN.A = self.A.copy() if self.A is not None else None

        FN.tms = self.tms.copy()
        FN.res = self.res.copy() if self.res is not None else None
        FN.err = self.err.copy() if self.err is not None else None

        FN.opts = self.opts.copy()

        return FN

    def prep(self):
        '''
        Construct function values on the spatial grid.
        '''

        if self.f is None or self.Y is not None: return

        self.tms['prep'] = time.time()

        if self.with_tt:
            self.res = { 'evals': 0, 't_func': 0. }

            def func(ind):
                ind = ind.astype(int)
                X = self.SG.comp(ind.T)
                t = time.time()
                Y = self.f(X, ind.T) if self.opts['is_f_with_i'] else self.f(X)
                self.res['t_func']+= time.time() - t
                self.res['evals']+= ind.shape[0]
                return Y

            if self.opts['Y0'] is None:
                Z = tt.rand(self.SG.n, self.SG.n.shape[0], 1)
            else:
                Z = self.opts['Y0'].copy()

                rmax = None
                for n_, r_ in zip(self.SG.n, Z.r[1:]):
                    if r_ > n_ and (rmax is None or rmax > n_): rmax = n_
                if rmax is not None: Z = Z.round(rmax=rmax)

            try:
                log = open('./tmp.txt', 'w')
                stdout0 = sys.stdout
                sys.stdout = log

                self.Y = cross(func, Z, eps=self.eps,
                    nswp=self.opts['nswp'], kickrank=self.opts['kickrank'],
                    rf=self.opts['rf'], verbose=True
                )
            finally:
                log.close()
                sys.stdout = stdout0

            log = open('./tmp.txt', 'r')
            res = log.readlines()[-1].split('swp: ')[1]
            self.res['iters'] = int(res.split('/')[0])+1
            res = res.split('er_rel = ')[1]
            self.res['err_rel'] = float(res.split('er_abs = ')[0])
            res = res.split('er_abs = ')[1]
            self.res['err_abs'] = float(res.split('erank = ')[0])
            res = res.split('erank = ')[1]
            self.res['erank'] = float(res.split('fun_eval')[0])
            log.close()

            self.res['t_func']/= self.res['evals']
            self.tms['func'] = self.res['t_func']
        else:
            self.tms['func'] = time.time()

            X = self.SG.comp()
            if self.opts['is_f_with_i']:
                I = self.SG.comp(is_ind=True)
                self.Y = self.f(X, I)
            else:
                self.Y = self.f(X)
            self.Y = self.Y.reshape(self.SG.n, order='F')

            self.tms['func'] = (time.time() - self.tms['func']) / X.shape[1]

        self.tms['prep'] = time.time() - self.tms['prep']

    def calc(self):
        '''
        Build tensor of interpolation coefficients according to training data.
        '''

        if self.Y is None:
            s = 'Train data is not set. Can not prep. Call "init" before.'
            raise ValueError(s)

        self.tms['calc'] = time.time()

        if self.with_tt:
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
        else:
            self.A = self.Y.copy()

            for i in range(self.SG.d):
                n_ = self.SG.n.copy(); n_[[0, i]] = n_[[i, 0]]
                self.A = np.swapaxes(self.A, 0, i)
                self.A = self.A.reshape((self.SG.n[i], -1), order='F')
                self.A = Func.interpolate_cheb(self.A)
                self.A = self.A.reshape(n_, order='F')
                self.A = np.swapaxes(self.A, 0, i)

        self.tms['calc'] = time.time() - self.tms['calc']

    def comp(self, X, z=0.):
        '''
        Calculate values of interpolated function in given X points.

        INPUT:

        X - values of x variable
        type: ndarray (or list) [dimensions, number of points] of float

        z - value for points outside the interpolation limits
        type: float
        * If some point is not belongs to the interpolation limits, then
        * the corresponding value will be set to the z-value.

        OUTPUT:

        Y - approximated values of the function in given points
        type: ndarray [number of points] of float

        TODO: Vectorize calculations for points vector if possible.
        TODO: Add more accurate check for outer points if possible.
        '''

        if self.A is None:
            s = 'Interpolation is not done. Can not calc. Call "prep" before.'
            raise ValueError(s)

        if not isinstance(X, np.ndarray): X = np.array(X)

        def is_out(j):
            for i in range(self.SG.d):
                if X[i, j] < self.SG.l[i, 0]: return True
                if X[i, j] > self.SG.l[i, 1]: return True
            return False

        self.tms['comp'] = time.time()

        Y = np.ones(X.shape[1]) * z
        m = np.max(self.SG.n) - 1
        T = polynomials_cheb(X, m, self.SG.l)

        if self.with_tt:
            G = tt.tensor.to_list(self.A)

            for j in range(X.shape[1]):
                if is_out(j): continue

                Q = np.einsum('riq,i->rq', G[0], T[:self.SG.n[0], 0, j])
                for i in range(1, X.shape[0]):
                    Q = Q @ np.einsum('riq,i->rq', G[i], T[:self.SG.n[i], i, j])

                Y[j] = Q[0, 0]
        else:
            for j in range(X.shape[1]):
                if is_out(j): continue

                Q = self.A.copy()
                for i in range(X.shape[0]):
                    Q = np.tensordot(Q, T[:Q.shape[0], i, j], axes=([0], [0]))

                Y[j] = Q

        self.tms['comp'] = (time.time() - self.tms['comp']) / X.shape[1]

        return Y

    def info(self, n_test=None, is_print=True):
        '''
        Present info about interpolation result, including error check.

        INPUT:

        n_test - number of points for error check
        type1: None
        type2: int, >= 0
        * If set (is not None and is greater than zero) and interpolation is
        * ready, then interpolation result will be checked on a set
        * of random points from the uniform distribution with proper limits.

        is_print - flag:
            True  - print string info
            False - return string info
        type: bool

        OUTPUT:

        s - (if is_print == False) string with info
        type: str
        '''

        if self.A is not None and self.f is not None and n_test:
            self.test(n_test)


        s = '------------------ Function\n'
        s+= 'Format           : %1dD, %s\n'%(self.SG.d, 'TT, eps= %8.2e'%self.eps if self.with_tt else 'NP')

        s+= '------------------ Time\n'
        s+= 'Prep             : %8.2e sec. \n'%self.tms['prep']
        s+= 'Calc             : %8.2e sec. \n'%self.tms['calc']
        s+= 'Comp (average)   : %8.2e sec. \n'%self.tms['comp']
        s+= 'Func (average)   : %8.2e sec. \n'%self.tms['func']

        if self.with_tt:
            s+= '------------------ Cross appr. parameters \n'
            s+= 'nswp             : %8d \n'%self.opts['nswp']
            s+= 'kickrank         : %8d \n'%self.opts['kickrank']
            s+= 'rf               : %8.2e \n'%self.opts['rf']

        if self.with_tt and self.res is not None:
            s+= '------------------ Cross appr. result \n'
            s+= 'Func. evaluations: %8d \n'%self.res['evals']
            s+= 'Cross iterations : %8d \n'%self.res['iters']
            s+= 'Av. tt-rank      : %8.2e \n'%self.res['erank']
            s+= 'Cross err (rel)  : %8.2e \n'%self.res['err_rel']
            s+= 'Cross err (abs)  : %8.2e \n'%self.res['err_abs']

        if self.A is not None and self.f is not None and n_test:
            s+= '------------------ Test for uniform random points \n'
            s+= 'Number of points : %8d \n'%n_test
            s+= 'Error (max)      : %8.2e \n'%np.max(self.err)
            s+= 'Error (mean)     : %8.2e \n'%np.mean(self.err)
            s+= 'Error (min)      : %8.2e \n'%np.min(self.err)

        if not s.endswith('\n'):
            s+= '\n'
        if is_print:
            print(s[:-1])
        else:
            return s

    def test(self, n=100):
        '''
        Calculate interpolation error on a set of random points from the
        uniform distribution with proper limits.

        INPUT:

        n - number of points for error check
        type1: None
        type2: int, >= 0
        * If set (is not None and is greater than zero) and interpolation is
        * ready, then interpolation result will be checked on a set
        * of random points from the uniform distribution with proper limits.

        OUTPUT:

        err - absolute value of relative error for the generated random points
        type: np.ndarray [n] of float

        TODO: Add support for absolute error.
        '''

        if self.f is None:
            s = 'Function for interpolation is not set. Can not test'
            raise ValueError(s)

        if self.opts['is_f_with_i']:
            s = 'Function for interpolation requires indices. Can not test'
            raise ValueError(s)

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
        * It may be of type1 or type2 only in the case of only one function .

        OUTPUT:

        A - constructed matrix of coefficients
        type: ndarray [number of points, number of functions] of float

        LINKS:

        - https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform

        - https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/23972/versions/22/previews/chebfun/examples/approx/html/ChebfunFFT.html
        '''

        if not isinstance(Y, np.ndarray): Y = np.array(Y)
        if len(Y.shape) == 1: Y = Y.reshape(-1, 1)

        n = Y.shape[0]
        V = np.vstack([Y, Y[n-2 : 0 : -1, :]])
        A = np.fft.fft(V, axis=0).real
        A = A[:n, :] / (n - 1)
        A[0, :] /= 2.
        A[n-1, :] /= 2.

        return A
