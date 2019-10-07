import sys
import time
import numpy as np
import scipy as sp
from scipy.linalg import toeplitz as _toeplitz

import tt
from tt.cross.rectcross import cross as _cross

from . import polynomials_cheb

class Intertrain(object):
    '''
    Class for the fast multidimensional function interpolation
    by Chebyshev polynomials in the dense (numpy) or sparse
    (tensor train (TT) with cross approximation) format
    using Fast Fourier Transform (FFT).

    Basic usage:
    1 Initialize class instance with grid and accuracy parameters.
    2 Call "init" with function of interest as an argument.
    3 Call "prep" for interpolation process.
    4 Call "calc" for approximation of function values on given points.
    5 Call "info" for demonstration of calculation results.

    Advanced usage:
    - Call "grid" for the full Chebyshev grid.
    - Call "pois" for subset of Chebyshev grid for given indices.
    - Call "dif1" for the 1th order differentiation matrix on Chebyshev grid.
    - Call "dif2" for the 2th order differentiation matrix on Chebyshev grid.
    - Call "copy" to obtain new instance with the same parameters and results.
    - Call "test" to obtain interpolation accuracy for set of random points.
    '''

    def __init__(self, GR, eps=1.E-6, log_path='./tmp.txt', with_tt=True):
        '''
        INPUT:

        GR - the spatial grid
        type: Grid

        n - total number of points for each dimension
        type: ndarray (or list) [dimensions] of int, >= 2

        l - min-max values of variable for each dimension
        type: ndarray (or list) [dimensions, 2] of float, [:, 0] < [:, 1]
        * Note that [:, 0] are min and [:, 1] are max values for each dimension.

        eps - is the desired accuracy of the approximation
        type: float, > 0
        * Is used in tt-round and cross approximation operations.

        log_path - file for output of the computation info
        type: str
        * In the current version it is used only for cross appr. output.

        with_tt - flag:
            True  - sparse (TT) format will be used
            False - dense (numpy) format will be used
        type: bool
        '''

        self.init(GR, eps, log_path, with_tt)

    def init(self, GR, eps=1.E-6, log_path='./tmp.txt', with_tt=True):
        '''
        Init all parameters of the class instance.

        INPUT: see Intertrain.__init__

        OUTPUT:

        self - class instance
        type: Intertrain
        '''

        self.GR = GR
        self.eps = eps
        self.log_path = log_path
        self.with_tt = with_tt

    def copy(self, is_full=True):
        '''
        Create a copy of the class instance.

        INPUT:

        is_full - flag:
            True  - interpolation result (if exists) will be also copied
            False - only calculation parameters will be copied
        type: bool

        OUTPUT:

        IT - new class instance
        type: Intertrain

        TODO! Add argument with changed opts (eps etc).
        '''

        IT = Intertrain(self.GR.copy(), self.eps, self.log_path, self.with_tt)

        if not is_full: return IT

        IT._t_init = self._t_init
        IT._t_prep = self._t_prep
        IT._t_calc = self._t_calc
        IT._t_func = self._t_func

        IT.err = self.err.copy() if self.err is not None else None
        IT.crs_res = self.crs_res.copy() if self.crs_res is not None else None

        IT.opts = self.opts.copy()

        IT.f = self.f
        IT.Y = self.Y.copy() if self.Y is not None else None
        IT.A = self.A.copy() if self.A is not None else None

        IT.D1 = self.D1.copy() if self.D1 is not None else None
        IT.D2 = self.D2.copy() if self.D2 is not None else None

        return IT

    def init(self, f=None, Y=None, opts=None):
        '''
        Init main calculation parameters and training data, using
        given function f (cross approximation will be applied if TT-format
        is used) or given tensor Y (in numpy or TT-format) of data.

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
        type: ndarray or TT-tensor [*n] of float
        * Should be in the TT-format if self.with_tt flag is set.
        * If is set, then function f will not be used for train data collection.

        opts - dictionary with optional parameters
        type: dict with fields:

            is_f_with_i - flag:
                True  - while function call the second argument with points
                        indices will be provided
                False - only points will be provided
            default: False
            type: bool

            nswp - for cross approximation
            default: 200
            type: int, > 0

            kickrank - for cross approximation
            default: 1
            type: int, > 0

            rf - for cross approximation
            default: 2
            type: int, > 0

            Y0 - initial guess for cross approximation
            default: None (random TT-tensor will be used)
            type: ndarray or TT-tensor [*n] of float

        OUTPUT:

        self - class instance
        type: Intertrain
        '''

        self.tms = {      # Saved times (durations)
            'init': 0.,   # training data collection
            'prep': 0.,   # interpolation coefficients calculation
            'calc': 0.,   # interpolation evaluation (average time for 1 poi)
            'func': 0.,   # function evaluation (average time for 1 poi)
        }

        self.err = None
        self.crs_res = None

        self.opts = {
            'is_f_with_i': (opts or {}).get('is_f_with_i', False),
            'nswp': (opts or {}).get('nswp', 200),
            'kickrank': (opts or {}).get('kickrank', 1),
            'rf': (opts or {}).get('rf', 2),
            'Y0': (opts or {}).get('Y0', None),
        }

        self.f = f
        self.Y = Y
        self.A = None

        self.D1 = None
        self.D2 = None

        if self.f is None or self.Y is not None: return self

        self._t_init = time.time()

        if self.with_tt:
            self.Y, self.crs_res = Intertrain.cross(
                self.f, self.pois, self.n, self.eps, self.opts, self.log_path
            )
            self._t_func = self.crs_res['t_func']
        else:
            self._t_func = time.time()

            X = self.grid()
            if self.opts['is_f_with_i']:
                I = self.grid(is_ind=True)
                self.Y = self.f(X, I)
            else:
                self.Y = self.f(X)
            self.Y = self.Y.reshape(self.n, order='F')

            self._t_func = (time.time() - self._t_func) / X.shape[1]

        self._t_init = time.time() - self._t_init

        return self

    def prep(self):
        '''
        Build tensor of interpolation coefficients according to training data.

        OUTPUT:

        self - class instance
        type: Intertrain
        '''

        if self.Y is None:
            s = 'Train data is not set. Can not prep. Call "init" before.'
            raise ValueError(s)

        self._t_prep = time.time()

        if self.with_tt:
            G = tt.tensor.to_list(self.Y)

            for i in range(self.d):
                G_sh = G[i].shape
                G[i] = np.swapaxes(G[i], 0, 1)
                G[i] = G[i].reshape((G_sh[1], -1))
                G[i] = Intertrain.interpolate(G[i])
                G[i] = G[i].reshape((G_sh[1], G_sh[0], G_sh[2]))
                G[i] = np.swapaxes(G[i], 0, 1)

            self.A = tt.tensor.from_list(G)
            self.A = self.A.round(self.eps)
        else:
            self.A = self.Y.copy()

            for i in range(self.d):
                n_ = self.n.copy(); n_[[0, i]] = n_[[i, 0]]
                self.A = np.swapaxes(self.A, 0, i)
                self.A = self.A.reshape((self.n[i], -1), order='F')
                self.A = Intertrain.interpolate(self.A)
                self.A = self.A.reshape(n_, order='F')
                self.A = np.swapaxes(self.A, 0, i)

        self._t_prep = time.time() - self._t_prep

        return self

    def calc(self, X, z=0.):
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
            for i in range(self.d):
                if X[i, j] < self.l[i, 0] or X[i, j] > self.l[i, 1]: return True
            return False

        self._t_calc = time.time()

        Y = np.ones(X.shape[1]) * z
        m = np.max(self.n) - 1
        T = polynomials_cheb(X, m, self.l)

        if self.with_tt:
            G = tt.tensor.to_list(self.A)

            for j in range(X.shape[1]):
                if is_out(j): continue

                Q = np.einsum('riq,i->rq', G[0], T[:self.n[0], 0, j])
                for i in range(1, X.shape[0]):
                    Q = Q @ np.einsum('riq,i->rq', G[i], T[:self.n[i], i, j])

                Y[j] = Q[0, 0]
        else:
            for j in range(X.shape[1]):
                if is_out(j): continue

                Q = self.A.copy()
                for i in range(X.shape[0]):
                    Q = np.tensordot(Q, T[:Q.shape[0], i, j], axes=([0], [0]))

                Y[j] = Q

        self._t_calc = (time.time() - self._t_calc) / X.shape[1]

        return Y

    def info(self, n=None):
        '''
        Present info about interpolation result, including error check.

        INPUT:

        n - number of points for error check
        * See Intertrain.test function.

        TODO! Cut lim info if too many dimensions?
        TODO! Replace prints by string construction.
        '''

        print('------------------ Intertain')
        print('Format           : %1dD, %s'%(self.d, 'TT, eps= %8.2e'%self.eps if self.with_tt else 'NP'))

        print('------------------ Time')
        print('Init             : %8.2e sec. '%self.tms['init'])
        print('Prep             : %8.2e sec. '%self.tms['prep'])
        print('Calc (average)   : %8.2e sec. '%self.tms['calc'])
        print('Func (average)   : %8.2e sec. '%self.tms['func'])

        if self.with_tt:
            print('------------------ Cross appr. parameters')
            print('nswp             : %8d'%self.opts['nswp'])
            print('kickrank         : %8d'%self.opts['kickrank'])
            print('rf               : %8.2e'%self.opts['rf'])

        if self.with_tt and self.crs_res is not None:
            print('------------------ Cross appr. result')
            print('Func. evaluations: %8d'%self.crs_res['evals'])
            print('Cross iterations : %8d'%self.crs_res['iters'])
            print('Av. tt-rank      : %8.2e'%self.crs_res['erank'])
            print('Cross err (rel)  : %8.2e'%self.crs_res['err_rel'])
            print('Cross err (abs)  : %8.2e'%self.crs_res['err_abs'])

        if self.A is not None and self.f is not None and n:
            self.test(n)
            print('------------------ Test for uniform random points')
            print('Number of points : %8d'%n)
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

        X_rand = self.GR.rand(n)
        u_calc = self.calc(X_rand)
        u_real = self.f(X_rand)
        self.err = np.abs((u_real - u_calc) / u_real)

        return self.err

    def dif1(self):
        '''
        Construct the first order differentiation matrix on Chebyshev grid
        on interval of the first spatial variable.

        OUTPUT:

        D1 - first order differentiation matrix
        type: ndarray [n[0], n[0]] of float
        '''

        l_min, l_max = self.l[0, 0], self.l[0, 1]
        self.D1 = Intertrain.chdiv(self.n[0], 1, l_min, l_max)[0]
        return self.D1

    def dif2(self):
        '''
        Construct the second order differentiation matrix on Chebyshev grid
        on interval of the first spatial variable.

        OUTPUT:

        D2 - second order differentiation matrix
        type: ndarray [n[0], n[0]] of float
        '''

        l_min, l_max = self.l[0, 0], self.l[0, 1]
        self.D1, self.D2 = Intertrain.chdiv(self.n[0], 2, l_min, l_max)
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

    @staticmethod
    def chdiv(n, m, l_min=-1., l_max=+1.):
        '''
        Construct differentiation matrices on the Chebyshev grid
        of order 1, 2, ..., m and size n x n on interval [l_min, l_max].

        INPUT:

        n - matrix size
        type: int, >= 2

        m - maximum matrix order (will construct for 1, 2, ..., m)
        type: int, >= 1

        l_min - min value of variable
        type: float, < l_max

        l_max - max value of variable
        type: float, > l_min

        OUTPUT:

        D - list of differentiation matrices (D1, D2, ..., Dm)
        type: tuple [m] of ndarray [n, n] of float

        TODO! Check and update the code.
        '''

        n1 = np.int(np.floor(n / 2))
        n2 = np.int(np.ceil(n / 2))
        k = np.arange(n)
        th = k * np.pi / (n - 1)

        T = np.tile(th/2, (n, 1))
        DX = 2. * np.sin(T.T + T) * np.sin(T.T - T)
        DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))
        DX[range(n), range(n)] = 1.
        DX = DX.T

        Z = 1. / DX
        Z[range(n), range(n)] = 0.

        C = _toeplitz((-1.)**k)
        C[+0, :]*= 2
        C[-1, :]*= 2
        C[:, +0]*= 0.5
        C[:, -1]*= 0.5

        D_list = []
        D = np.eye(n)
        l = 2. / (l_max - l_min)
        for i in range(m):
            D = (i + 1) * Z * (C * np.tile(np.diag(D), (n, 1)).T - D)
            D[range(n), range(n)] = -np.sum(D, axis=1)
            D_list.append(D * l)
            l = l * l

        return tuple(D_list)

    @staticmethod
    def cross(f, f_pois, n, eps=1.E-6, opts=None, fpath='./tmp.txt'):
        '''
        Construct tensor of function values on given tensor product
        grid by the cross approximation method in the TT-format.

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

        f_pois - function that calculate grid points for given indices
        type: function
            inp:
                I - indices of X points
                type: ndarray [dimensions, number of points] of int
            out:
                X - the corresponding grid points
                type: ndarray [dimensions, number of points] of float

        n - total number of points for each dimension
        type: ndarray (or list) [dimensions] of int, >= 2

        eps - is the desired accuracy of the approximation
        type: float, > 0

        opts - dictionary with optional parameters
        type: dict with fields:

            is_f_with_i - flag:
                True  - while function call the second argument with points
                        indices will be provided
                False - only points will be provided
            default: False
            type: bool

            nswp - for cross approximation
            default: 200
            type: int, > 0

            kickrank - for cross approximation
            default: 1
            type: int, > 0

            rf - for cross approximation
            default: 2
            type: int, > 0

            Y0 - initial guess for cross approximation
            default: None (random TT-tensor will be used)
            type: ndarray or TT-tensor [*n] of float

        fpath - file path for output cross approximation native info
        type: str

        OUTPUT: /tuple/

        Y - tensor of function values on given tensor product grid
        type: TT-tensor [*n] of float

        crs_res - dictionary with calculation details
        type: dict
        * Fields: evals (int), iters (int), err_rel (float), err_abs (float),
        * erank (float), t_func (float).
        '''

        if not isinstance(n, np.ndarray): n = np.array(n)

        opts = {
            'is_f_with_i': (opts or {}).get('is_f_with_i', False),
            'nswp': (opts or {}).get('nswp', 200),
            'kickrank': (opts or {}).get('kickrank', 1),
            'rf': (opts or {}).get('rf', 2),
            'Y0': (opts or {}).get('Y0', None),
        }

        crs_res = { 'evals': 0, 't_func': 0. }

        def func(ind):
            ind = ind.astype(int)
            X = f_pois(ind.T)
            t = time.time()
            Y = f(X, ind.T) if opts['is_f_with_i'] else f(X)
            crs_res['t_func']+= time.time() - t
            crs_res['evals']+= ind.shape[0]
            return Y

        if opts['Y0'] is None:
            Z = tt.rand(n, n.shape[0], 1)
        else:
            Z = opts['Y0'].copy()

            rmax = None
            for n_, r_ in zip(n, Z.r[1:]):
                if r_ > n_ and (rmax is None or rmax > n_): rmax = n_
            if rmax is not None: Z = Z.round(rmax=rmax)

        try:
            log = open(fpath, 'w')
            stdout0 = sys.stdout
            sys.stdout = log

            Y = _cross(
                func, Z, eps=eps, nswp=opts['nswp'], kickrank=opts['kickrank'],
                rf=opts['rf'], verbose=True
            )
        finally:
            log.close()
            sys.stdout = stdout0

        log = open(fpath, 'r')
        res = log.readlines()[-1].split('swp: ')[1]
        crs_res['iters'] = int(res.split('/')[0])+1
        res = res.split('er_rel = ')[1]
        crs_res['err_rel'] = float(res.split('er_abs = ')[0])
        res = res.split('er_abs = ')[1]
        crs_res['err_abs'] = float(res.split('erank = ')[0])
        res = res.split('erank = ')[1]
        crs_res['erank'] = float(res.split('fun_eval')[0])
        log.close()

        crs_res['t_func']/= crs_res['evals']

        return (Y, crs_res)
