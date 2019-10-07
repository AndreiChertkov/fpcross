import sys
import time
import numpy as np

import tt
from tt.cross.rectcross import cross

from . import polynomials_cheb
from . import dif_cheb

class Func(object):
    '''
    Class for the fast multidimensional function interpolation
    by Chebyshev polynomials in the dense (numpy) or sparse
    (tensor train (TT) with cross approximation) format
    using Fast Fourier Transform (FFT).

    Basic usage:
    1 Initialize class instance with grid and accuracy parameters.
    2 Call "init" with function of interest and interpolation options.
    3 Call "prep" for construction of the function on the spatial grid.
    3 Call "calc" for interpolation process.
    4 Call "comp" for approximation of function values on given points.
    6 Call "info" for demonstration of calculation results.

    Advanced usage:
    - Call "dif1" for the 1th order differentiation matrix on Chebyshev grid.
    - Call "dif2" for the 2th order differentiation matrix on Chebyshev grid.
    - Call "copy" to obtain new instance with the same parameters and results.
    - Call "test" to obtain interpolation accuracy for set of random points.
    '''

    def __init__(self, GR, eps=1.E-6, log_path='./tmp.txt', with_tt=True):
        '''
        INPUT:

        GR - spatial grid for function interpolation
        type: fpcross.Grid

        eps - is the desired accuracy of the approximation
        type: float, > 0
        * Is used in tt-round and tt-cross approximation operations.

        log_path - file for output of the computation info
        type: str
        * In the current version it is used only for cross approximation output.

        with_tt - flag:
            True  - sparse (tensor train, TT) format will be used
            False - dense (numpy, NP) format will be used
        type: bool
        '''

        self.GR = GR
        self.eps = eps
        self.log_path = log_path
        self.with_tt = with_tt

        self.opts = {}
        self.init()

    def init(self, f=None, Y=None, opts={}):
        '''
        Init the main parameters of the class instance.

        INPUT:

        f - function that calculate tensor elements for given points
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
        type1: ndarray [*n] of float
        type2: TT-tensor [*n] of float
        * Should be in the TT-format (type2) if self.with_tt flag is set.
        * If is set, then function f will not be used for train data collection,
        * but may be used in self.test for check of the interpolation result.

        opts - dictionary with optional parameters
        type: dict
        fld:is_f_with_i - flag:
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

        OUTPUT:

        self - class instance
        type: fpcross.Func

        TODO! Check case for update with Y0 opt changed to None.
        '''

        self.f = f
        self.Y = Y
        self.A = None

        self.D1 = None
        self.D2 = None

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

        return self

    def copy(self, is_full=True):
        '''
        Create a copy of the class instance.

        INPUT:

        is_full - flag:
            True  - interpolation result (if exists) will be also copied
            False - only parameters will be copied
        type: bool

        OUTPUT:

        IT - new class instance
        type: fpcross.Func

        TODO! Add argument with changed opts (eps etc).
        '''

        IT = Func(self.GR.copy(), self.eps, self.log_path, self.with_tt)

        if not is_full: return IT

        IT.f = self.f
        IT.Y = self.Y.copy() if self.Y is not None else None
        IT.A = self.A.copy() if self.A is not None else None

        IT.D1 = self.D1.copy() if self.D1 is not None else None
        IT.D2 = self.D2.copy() if self.D2 is not None else None

        IT.tms = self.tms.copy()
        IT.res = self.res.copy() if self.res is not None else None
        IT.err = self.err.copy() if self.err is not None else None

        IT.opts = self.opts.copy()

        return IT

    def prep(self):
        '''
        Construct function values on the spatial grid.

        OUTPUT:

        self - class instance
        type: fpcross.Func
        '''

        if self.f is None or self.Y is not None: return self

        self.tms['prep'] = time.time()

        if self.with_tt:
            self.res = { 'evals': 0, 't_func': 0. }

            def func(ind):
                ind = ind.astype(int)
                X = self.GR.comp(ind.T)
                t = time.time()
                Y = self.f(X, ind.T) if self.opts['is_f_with_i'] else self.f(X)
                self.res['t_func']+= time.time() - t
                self.res['evals']+= ind.shape[0]
                return Y

            if self.opts['Y0'] is None:
                Z = tt.rand(self.GR.n, self.GR.n.shape[0], 1)
            else:
                Z = self.opts['Y0'].copy()

                rmax = None
                for n_, r_ in zip(self.GR.n, Z.r[1:]):
                    if r_ > n_ and (rmax is None or rmax > n_): rmax = n_
                if rmax is not None: Z = Z.round(rmax=rmax)

            try:
                log = open(self.log_path, 'w')
                stdout0 = sys.stdout
                sys.stdout = log

                self.Y = cross(func, Z, eps=self.eps,
                    nswp=self.opts['nswp'], kickrank=self.opts['kickrank'],
                    rf=self.opts['rf'], verbose=True
                )
            finally:
                log.close()
                sys.stdout = stdout0

            log = open(self.log_path, 'r')
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

            X = self.GR.comp()
            if self.opts['is_f_with_i']:
                I = self.GR.comp(is_ind=True)
                self.Y = self.f(X, I)
            else:
                self.Y = self.f(X)
            self.Y = self.Y.reshape(self.GR.n, order='F')

            self.tms['func'] = (time.time() - self.tms['func']) / X.shape[1]

        self.tms['prep'] = time.time() - self.tms['prep']

        return self

    def calc(self):
        '''
        Build tensor of interpolation coefficients according to training data.

        OUTPUT:

        self - class instance
        type: fpcross.Func
        '''

        if self.Y is None:
            s = 'Train data is not set. Can not prep. Call "init" before.'
            raise ValueError(s)

        self.tms['calc'] = time.time()

        if self.with_tt:
            G = tt.tensor.to_list(self.Y)

            for i in range(self.GR.d):
                G_sh = G[i].shape
                G[i] = np.swapaxes(G[i], 0, 1)
                G[i] = G[i].reshape((G_sh[1], -1))
                G[i] = Func.interpolate(G[i])
                G[i] = G[i].reshape((G_sh[1], G_sh[0], G_sh[2]))
                G[i] = np.swapaxes(G[i], 0, 1)

            self.A = tt.tensor.from_list(G)
            self.A = self.A.round(self.eps)
        else:
            self.A = self.Y.copy()

            for i in range(self.GR.d):
                n_ = self.GR.n.copy(); n_[[0, i]] = n_[[i, 0]]
                self.A = np.swapaxes(self.A, 0, i)
                self.A = self.A.reshape((self.GR.n[i], -1), order='F')
                self.A = Func.interpolate(self.A)
                self.A = self.A.reshape(n_, order='F')
                self.A = np.swapaxes(self.A, 0, i)

        self.tms['calc'] = time.time() - self.tms['calc']

        return self

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

        TODO! Vectorize calculations for points vector if possible.
        TODO! Add more accurate check for outer points if possible.
        '''

        if self.A is None:
            s = 'Interpolation is not done. Can not calc. Call "prep" before.'
            raise ValueError(s)

        if not isinstance(X, np.ndarray): X = np.array(X)

        def is_out(j):
            for i in range(self.GR.d):
                if X[i, j] < self.GR.l[i, 0]: return True
                if X[i, j] > self.GR.l[i, 1]: return True
            return False

        self.tms['comp'] = time.time()

        Y = np.ones(X.shape[1]) * z
        m = np.max(self.GR.n) - 1
        T = polynomials_cheb(X, m, self.GR.l)

        if self.with_tt:
            G = tt.tensor.to_list(self.A)

            for j in range(X.shape[1]):
                if is_out(j): continue

                Q = np.einsum('riq,i->rq', G[0], T[:self.GR.n[0], 0, j])
                for i in range(1, X.shape[0]):
                    Q = Q @ np.einsum('riq,i->rq', G[i], T[:self.GR.n[i], i, j])

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

    def info(self, n_test=None):
        '''
        Present info about interpolation result, including error check.

        INPUT:

        n_test - number of points for error check
        type1: None
        type2: int, >= 0
        * If set (is not None and is greater than zero) and interpolation is
        * ready, then interpolation result will be checked on a set
        * of random points from the uniform distribution with proper limits.

        TODO! Cut lim info if too many dimensions?
        TODO! Replace prints by string construction.
        '''

        if self.A is not None and self.f is not None and n_test:
            self.test(n_test)

        print('------------------ Intertain')
        print('Format           : %1dD, %s'%(self.GR.d, 'TT, eps= %8.2e'%self.eps if self.with_tt else 'NP'))

        print('------------------ Time')
        print('Prep             : %8.2e sec. '%self.tms['prep'])
        print('Calc             : %8.2e sec. '%self.tms['calc'])
        print('Comp (average)   : %8.2e sec. '%self.tms['comp'])
        print('Func (average)   : %8.2e sec. '%self.tms['func'])

        if self.with_tt:
            print('------------------ Cross appr. parameters')
            print('nswp             : %8d'%self.opts['nswp'])
            print('kickrank         : %8d'%self.opts['kickrank'])
            print('rf               : %8.2e'%self.opts['rf'])

        if self.with_tt and self.res is not None:
            print('------------------ Cross appr. result')
            print('Func. evaluations: %8d'%self.res['evals'])
            print('Cross iterations : %8d'%self.res['iters'])
            print('Av. tt-rank      : %8.2e'%self.res['erank'])
            print('Cross err (rel)  : %8.2e'%self.res['err_rel'])
            print('Cross err (abs)  : %8.2e'%self.res['err_abs'])

        if self.A is not None and self.f is not None and n_test:
            print('------------------ Test for uniform random points')
            print('Number of points : %8d'%n_test)
            print('Error (max)      : %8.2e'%np.max(self.err))
            print('Error (mean)     : %8.2e'%np.mean(self.err))
            print('Error (min)      : %8.2e'%np.min(self.err))

        print('------------------')

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

        TODO! Add support for absolute error.
        '''

        if self.f is None:
            s = 'Function for interpolation is not set. Can not test'
            raise ValueError(s)

        if self.opts['is_f_with_i']:
            s = 'Function for interpolation requires indices. Can not test'
            raise ValueError(s)

        X = self.GR.rand(n)
        u = self.f(X)
        v = self.comp(X)
        self.err = np.abs((u - v) / u)

        return self.err

    def dif1(self):
        '''
        Construct the first order differentiation matrix on Chebyshev grid
        on interval of the first spatial variable.

        OUTPUT:

        D1 - first order differentiation matrix
        type: ndarray [n[0], n[0]] of float
        '''

        n, l_min, l_max = self.GR.n[0], self.GR.l[0, 0], self.GR.l[0, 1]
        self.D1 = dif_cheb(n, 1, l_min, l_max)[0]
        return self.D1

    def dif2(self):
        '''
        Construct the second order differentiation matrix on Chebyshev grid
        on interval of the first spatial variable.

        OUTPUT:

        D2 - second order differentiation matrix
        type: ndarray [n[0], n[0]] of float
        '''

        n, l_min, l_max = self.GR.n[0], self.GR.l[0, 0], self.GR.l[0, 1]
        self.D1, self.D2 = dif_cheb(n, 2, l_min, l_max)
        return self.D2

    @staticmethod
    def interpolate(Y):
        '''
        Find coefficients a_i for interpolation of 1D functions by Chebyshev
        polynomials f(x) = \sum_{i} (a_i * T_i(x)) using Fast Fourier Transform.

        It can find coefficients for several functions on the one call,
        if all functions have equal numbers of grid points.

        INPUT:

        Y - values of function at the nodes of the Chebyshev grid
            x_j = cos(\pi j / (N - 1)), j = 0, 1 ,..., N-1,
            where N is a number of points
        type: ndarray (or list) [number of points, number of functions] of float
        * In the case of only one function,
        * it may be ndarray (or list) [number of points].

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
