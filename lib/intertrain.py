import sys
import time

import numpy as np
from scipy.linalg import toeplitz

import tt
from tt.cross.rectcross import cross as rect_cross

class Intertrain(object):
    '''
    Class for the fast multidimensional function interpolation
    by Chebyshev polynomials in the dense (numpy) or sparse
    (tensor train (TT) with cross approximation) format
    using Fast Fourier Transform (FFT).

    Basic usage:
    1 Initialize class instance with grid and accuracy parameters.
    2 Call "init" with function of interest as argument.
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

    def __init__(self, n, l, ns=0, eps=1.E-6, log_path='./tmp.txt', with_tt=True):
        '''
        INPUT:

        n - total number of points for each dimension
        type: ndarray (or list) [dimensions] of int >= 2

        l - min-max values of variable for each dimension
        type: ndarray (or list) [dimensions, 2] of float

        ns - (optional) is a number of dimensions that should be skipped
        (are not transformed while interpolation)
        * Is not supported in the current version!!!
        type: int 0 <= and <= dimensions

        eps - (optional) is the desired accuracy of the approximation
        (is used in tt-round and cross approximation operations)
        type: float > 0

        log_path - (optional) file for output computation info
        * In the current version it is used only for cross appr. output
        type: str

        with_tt - (optional) flag; if True, then TT-format will be used
        instead of dense (numpy) format
        type: bool
        '''

        self.n = n if isinstance(n, np.ndarray) else np.array(n)
        self.l = l if isinstance(l, np.ndarray) else np.array(l)
        self.d = self.l.shape[0]
        self.ns = ns
        self.eps = eps
        self.log_path = log_path
        self.with_tt = with_tt

        if self.n.shape[0] != self.l.shape[0]:
            raise IndexError('Shape mismatch for n and l params.')

        self.init()

    def copy(self, is_full=True):
        '''
        Create a copy of the class instance.

        INPUT:

        is_full - (optional) flag; if True, then interpolation result
        (if exists) will be also copied
        type: bool

        OUTPUT:

        IT - new class instance
        type: Intertrain
        '''

        IT = Intertrain(self.n, self.l, self.ns, self.eps, self.log_path, self.with_tt)

        if not is_full:
            return IT

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

        f - (optional) function that calculate tensor elements for given points
        (if ns>0, then first ns dimensions should be simple integer indices)
        type: function
            inp:
                X - points for calculation
                type: ndarray [dimensions, number of points] of float,
                I - indices of X points (opts.is_f_with_i flag is set)
                type: ndarray [dimensions, number of points] of int,
            out:
                Y - function values on X points
                type: ndarray [number of points] of float

        Y - (optional) tensor of function values on nodes of the Chebyshev grid
        (for different axes different numbers of points may be used)
        type: ndarray or TT-tensor [*n] of float

        opts - (optional) dictionary with optional parameters:
            is_f_with_i - (default: False) if True, then while function call the second argument with points indices will be provided
            nswp - (default: 200) for cross approximation
            kickrank - (default: 1) for cross approximation
            rf - (default: 2) for cross approximation
        type: dict

        OUTPUT:

        self - class instance
        type: Intertrain
        '''

        self._t_init = 0. # Time of training data collection
        self._t_prep = 0. # Time of interpolation coefficients calculation
        self._t_calc = 0. # Average time of interpolation evaluation for 1 poi
        self._t_func = 0. # Average time of function evaluation for 1 poi

        self.err = None
        self.crs_res = None

        self.opts = opts or {}

        self.f = f
        self.Y = Y
        self.A = None

        self.D1 = None
        self.D2 = None

        if self.f is None or self.Y is not None:
            return self

        self._t_init = time.time()

        if self.with_tt:
            self.Y, self.crs_res = Intertrain.cross(
                self.f, self.pois, self.n, self.eps, self.opts, self.log_path)
            self._t_func = self.crs_res['t_func']
        else:
            self._t_func = time.time()
            X = self.grid()
            if self.opts.get('is_f_with_i') == True:
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
            raise ValueError('Train data is not set. Call "init" before.')

        self._t_prep = time.time()

        if self.with_tt:
            G = tt.tensor.to_list(self.Y)

            for i in range(self.ns, self.Y.d):
                sh = G[i].shape
                G[i] = np.swapaxes(G[i], 0, 1)
                G[i] = G[i].reshape((sh[1], -1))
                G[i] = Intertrain.interpolate(G[i])
                G[i] = G[i].reshape((sh[1], sh[0], sh[2]))
                G[i] = np.swapaxes(G[i], 0, 1)

            self.A = tt.tensor.from_list(G)
            self.A = self.A.round(self.eps)
        else:
            self.A = self.Y.copy()
            for i in range(len(self.Y.shape)):
                self.A = np.swapaxes(self.A, 0, i)
                self.A = self.A.reshape((self.Y.shape[i], -1))
                self.A = Intertrain.interpolate(self.A, is_real=False)
                self.A = self.A.reshape(self.Y.shape)
                self.A = np.swapaxes(self.A, i, 0)

        self._t_prep = time.time() - self._t_prep

        return self

    def calc(self, X):
        '''
        Calculate values of interpolated function in given X points.

        INPUT:

        X - values of x variable
        type: ndarray (or list) [dimensions, number of points] of float

        OUTPUT:

        Y - approximated values of the function in given points
        type: ndarray [number of points] of float

        TODO! Vectorize calculations for points vector
        '''

        if self.A is None:
            raise ValueError('Interpolation is not done. Call "prep" before.')

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self._t_calc = time.time()

        if self.with_tt:
            Y = np.zeros(X.shape[1])
            G = tt.tensor.to_list(self.A)
            r = max([G_.shape[1] for G_ in G])
            T = Intertrain.polynomials(X, r, self.l)

            for j in range(X.shape[1]):
                i = 0

                if i <= self.ns - 1:
                    Q = G[i][:, int(X[i, j]), :]
                else:
                    F = T[:G[i].shape[1], i, j]
                    Q = np.tensordot(G[i], F, axes=([1], [0]))

                for i in range(1, X.shape[0]):
                    if i <= self.ns - 1:
                        Q = np.dot(Q, G[i][:, int(X[i, j]), :])
                    else:
                        F = T[:G[i].shape[1], i, j]
                        Q = np.dot(Q, np.tensordot(G[i], F, axes=([1], [0])))

                Y[j] = Q[0, 0]
        else:
            Y = np.zeros(X.shape[1])
            r = np.max(self.n-1)
            T = Intertrain.polynomials(X, r, self.l)

            for j in range(X.shape[1]):
                is_out = False
                for i in range(X.shape[0]):
                    if X[i, j] < self.l[i, 0] or X[i, j] > self.l[i, 1]:
                        is_out = True
                if is_out: continue

                B = self.A.real.copy()
                for i in range(X.shape[0]):
                    F = T[:B.shape[0], i, j]
                    B = np.tensordot(B, F, axes=([0], [0]))

                Y[j] = B

        self._t_calc = (time.time() - self._t_calc) / X.shape[1]

        return Y

    def calc_d1(self, X):
        if self.A is None:
            raise ValueError('Interpolation is not done. Call "prep" before.')

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        Y = np.zeros(X.shape[1])
        r = np.max(self.n-1)
        T = Intertrain.polynomials(X, r, self.l)

        for j in range(X.shape[1]):
            B = self.A.real.copy()
            for i in range(X.shape[0]):
                F = T[:B.shape[0], i, j]
                B = np.tensordot(B, F, axes=([0], [0]))

            Y[j] = B

        return Y

    def calc_d2(self, X):
        return self.calc(X)

    def info(self, f=None, npoi=10, a=-1., b=1.):
        '''
        Present info about interpolation result, including error check.

        INPUT:

        f - (optional) function that calculate tensor elements for given points
        (if not set, then function from init arg will be used)
        (if ns>0, then first ns dimensions should be simple integer indices)
        type: function
            inp type: ndarray [dimensions, number of points] of float
            out type: ndarray [number of points] of float

        npoi - (optional) number of points for error check
        type: int

        a - (optional) minimum value for random check point
        type: float

        b - (optional) maximum value for random check point
        type: float
        '''

        print('------------------ Parameters')
        print('Format           : %s'%('TT' if self.with_tt else 'NP'))
        print('Dimensions       : %8d'%self.d)

        if self.with_tt:
            print('Accuracy         : %8.2e'%self.eps)

        for i, [n, l] in enumerate(zip(self.n, self.l)):
            opts = (i+1, n, l[0], l[1])
            print('Dim %-2d | Poi %-3d | Min %-6.3f | Max %-6.3f |'%opts)

        if self.with_tt and self.crs_res is not None:
            print('------------------ Result (cross appr)')
            print('Func. evaluations: %8d'%self.crs_res['evals'])
            print('Cross iterations : %8d'%self.crs_res['iters'])
            print('Av. tt-rank      : %8.2e'%self.crs_res['erank'])
            print('Cross err (rel)  : %8.2e'%self.crs_res['err_rel'])
            print('Cross err (abs)  : %8.2e'%self.crs_res['err_abs'])

        print('------------------ Time')
        print('Init             : %8.2e sec. '%self._t_init)
        print('Prep             : %8.2e sec. '%self._t_prep)
        print('Calc (average)   : %8.2e sec. '%self._t_calc)
        print('Func (average)   : %8.2e sec. '%self._t_func)

        if self.A is not None and (f or self.f):
            self.test(f or self.f, npoi, a, b)
            print('------------------ Test (random points)')
            print('Number of points : %d'%npoi)
            print('Error (max)      : %8.2e '%np.max(self.err))
            print('Error (mean)     : %8.2e '%np.mean(self.err))
            print('Error (min)      : %8.2e '%np.min(self.err))

        print('------------------')

    def test(self, f=None, npoi=10, a=-1., b=1.):
        '''
        Calculate interpolation error on random points.

        INPUT:

        f - (optional) function that calculate tensor elements for given points
        (if not set, then function from init arg will be used)
        (if ns>0, then first ns dimensions should be simple integer indices)
        type: function
            inp type: ndarray [dimensions, number of points] of float
            out type: ndarray [number of points] of float

        npoi - (optional) number of points for error check
        type: int

        a - (optional) minimum value for random check point
        type: float

        b - (optional) maximum value for random check point
        type: float

        OUTPUT:

        err - absolute value of relative error in random points
        type: np.ndarray [npoi] of float
        '''

        f = f or self.f
        if f is None:
            raise ValueError('Function for interpolation is not set.')

        X = a + np.random.random((self.d, npoi)) * (b - a)
        u = f(X)
        v = self.calc(X)
        e = (v - u) / u
        self.err = np.abs(e)

        return self.err

    def pois(self, I):
        '''
        Get points of Chebyshev multidimensional (dim) grid
        for given multi indices I. Points for every axis k are calculated as
        x = cos(i[k]*pi/(n[k]-1)), where n[k] is a total number of points
        for selected axis k, and then these points are scaled according to
        given limits l.

        INPUT:

        I - multi index for the grid points
        type: ndarray (or list) [dimensions, number of points] of int

        OUTPUT:

        X - grid points
        type: ndarray [dimensions, number of points] of float
        '''

        if not isinstance(I, np.ndarray):
            I = np.array(I)

        n = np.repeat(self.n.reshape((-1, 1)), I.shape[1], axis=1)
        l1 = np.repeat(self.l[:, 0].reshape((-1, 1)), I.shape[1], axis=1)
        l2 = np.repeat(self.l[:, 1].reshape((-1, 1)), I.shape[1], axis=1)

        t = np.cos(np.pi * I / (n - 1))
        X = t * (l2 - l1) / 2.
        X+= (l2 + l1) / 2.

        return X

    def grid(self, is_ind=False):
        '''
        Get all points of flatten multidimensional Chebyshev grid
        (from max to min value).

        INPUT:

        is_ind - (optional) flag; if True, then indices of points will be
        returned; if false, then spatial grid points will be returned

        OUTPUT:

        I - (if is_ind == True) indices of grid points
        type: ndarray [dimensions, n1 * n2 * ... * nd] of int

        X - (if is_ind == False) grid points
        type: ndarray [dimensions, n1 * n2 * ... * nd] of float
        '''

        I = [np.arange(self.n[d]).reshape(1, -1) for d in range(self.d)]
        I = np.meshgrid(*I)
        I = np.array(I).reshape((self.d, -1))

        return I if is_ind else self.pois(I)

    def dif1(self):
        '''
        Construct first order differentiation matrix on Chebyshev grid
        on interval [-1, 1].

        OUTPUT:

        D1 - first order differentiation matrix
        type: ndarray [number of points, number of points] of float

        TODO! Add (check) support for custom limits.
        '''

        self.D1 = Intertrain.chdiv(self.n[0], 1)[0]
        return self.D1

    def dif2(self):
        '''
        Construct second order differentiation matrix on Chebyshev grid
        on interval [-1, 1].

        OUTPUT:

        D2 - second order differentiation matrix
        type: ndarray [number of points, number of points] of float

        TODO! Add (check) support for custom limits.
        '''

        self.D1, self.D2 = Intertrain.chdiv(self.n[0], 2)
        return self.D2

    @staticmethod
    def polynomials(X, m, l=None):
        '''
        Calculate Chebyshev polynomials of order 0, 1, ..., m in given X points.

        INPUT:

        X - values of x variable
        type: ndarray (or list) [dimensions, number of points] of float
        * in case of 1D, it may be ndarray (or list) [number of points]
        * in case of one point in 1D, it may be float
        * each value should be in range -1 <= >= 1

        m - max order of polynomial (function will calculate for 0, 1, ..., m)
        type: int, >= 0

        l - (optional) min-max values of variable for each dimension
        * default limits are [-1, 1]; if l is given then X values values will be
        * scaled from l-limits to [-1, 1] limits
        type: ndarray (or list) [dimensions, 2] of float

        OUTPUT:

        T - Chebyshev polynomials of order 0, 1, ..., m in given points
        type: ndarray [m+1, *X.shape] of float
        * if X is float, it will be ndarray [m+1]
        '''

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if l is not None:
            if not isinstance(l, np.ndarray):
                l = np.array(l)
            if len(X.shape) > 1:
                l1 = np.repeat(l[:, 0].reshape((-1, 1)), X.shape[-1], axis=1)
                l2 = np.repeat(l[:, 1].reshape((-1, 1)), X.shape[-1], axis=1)
            elif len(X.shape) == 1:
                l1 = np.repeat(l[0, 0], X.shape[0], axis=0)
                l2 = np.repeat(l[:, 1], X.shape[0], axis=0)
            else:
                l1 = l[0, 0]
                l2 = l[0, 1]
            X = (2. * X - l2 - l1)
            X/= (l2 - l1)

        if len(X.shape):
            T = np.ones([m+1] + list(X.shape))
        else:
            T = np.ones([m+1, 1])

        if m > 0:
            T[1,] = X.copy()
            for k in range(2, m+1):
                T[k, ] = 2. * X * T[k-1, ] - T[k-2, ]

        return T if len(X.shape) else T.reshape(-1)

    @staticmethod
    def interpolate(Y, is_real=True):
        '''
        Find coefficients a_i for interpolation of 1D functions by Chebyshev
        polynomials f(x) = \sum_{i} (a_i * T_i(x)) using Fast Fourier Transform.

        It can find coefficients for several functions on the one call
        if all functions have equal numbers of grid points.

        INPUT:

        Y - values of function at the nodes of the Chebyshev grid
        x_j=cos(\pi j/N), j = 0, 1, ..., N (N = number of points - 1)
        type: ndarray (or list) [number of points, number of functions] of float
        * in case of one function, it may be ndarray (or list) [number of points]

        OUTPUT:

        A - constructed matrix of coefficients
        type: ndarray [number of points, number of functions] of float

        LINKS:
        - https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform
        - https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/23972/versions/22/previews/chebfun/examples/approx/html/ChebfunFFT.html
        '''

        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        n = Y.shape[0]
        V = np.vstack([Y, Y[n-2:0:-1, :]])
        A = np.fft.fft(V, axis=0)
        if is_real:
            A = A.real
        A = A[:n, :] / (n - 1)
        A[0, :] /= 2.
        A[n-1, :] /= 2.
        return A

    @staticmethod
    def chdiv(n, m):
        '''
        Construct differentiation matrices on Chebyshev grid of order 1, 2, ..., m
        and size n x n on interval [-1, 1].

        INPUT:

        n - matrix size
        type: int, >= 2

        m - maximum matrix order (will construct for 1, 2, ..., m)
        type: int, >= 1

        OUTPUT:

        D - list of differentiation matrices (D1, D2, ..., Dm)
        type: [m] of ndarray [n, n] of float
        '''

        n1 = np.int(np.floor(n / 2))
        n2 = np.int(np.ceil(n / 2))
        k = np.arange(n)
        th = k * np.pi / (n-1)

        T = np.tile(th/2, (n, 1))
        DX = 2*np.sin(T.T+T)*np.sin(T.T-T)
        DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))
        DX[range(n), range(n)] = 1.
        DX = DX.T

        Z = 1 / DX
        Z[range(n), range(n)] = 0.

        C = toeplitz((-1.)**k)
        C[+0, :] *= 2
        C[-1, :] *= 2
        C[:, +0] *= 0.5
        C[:, -1] *= 0.5

        D_list = []
        D = np.eye(n)
        for ell in range(2):
            D = (ell + 1) * Z * (C * np.tile(np.diag(D), (n, 1)).T - D)
            D[range(n), range(n)] = -np.sum(D, axis=1)
            D_list.append(D)

        return tuple(D_list)

    @staticmethod
    def cross(f, f_pois, n, eps=1.E-6, opts=None, fpath='./tmp.txt'):
        '''
        Construct tensor in TT-format of function values on given tensor product
        grid by cross approximation.

        INPUT:

        f - function that calculate tensor elements for given points
        type: function
            inp type: ndarray [dimensions, number of points] of float
            output type: ndarray [number of points] of float

        f_pois - function that calculate grid points for given indices
        type: function
            inp type: ndarray [dimensions, number of points] of float
            output type: ndarray [dimensions, number of points] of float

        n - total number of points for each dimension
        type: ndarray (or list) [dimensions] of int >= 2

        eps - (optional) is the desired accuracy of the approximation
        type: float > 0

        opts - (optional) dictionary with optional cross aproximation parameters:
            nswp - (default: 200)
            kickrank - (default: 1)
            rf - (default: 2)
        type: dict

        OUTPUT:

        Y - tensor of function values on given tensor product grid
        type: TT-tensor [*n] of float

        crs_res - dictionary with calculation details
        * evals, iters, err_rel, err_abs, erank, t_func
        type: dict

        fpath - (optional) file path for output cross approximation native info
        type: str
        '''

        if not isinstance(n, np.ndarray): n = np.array(n)
        if opts is None: opts = {}

        crs_res = {}
        crs_res['evals'] = 0
        crs_res['t_func'] = 0.

        def func(ind):
            X = f_pois(ind.T)
            t = time.time()
            Y = f(X, ind.T) if opts.get('is_f_with_i') == True else f(X)
            crs_res['t_func'] += time.time() - t
            crs_res['evals'] += ind.shape[0]
            return Y

        log = open(fpath, 'w')
        stdout0 = sys.stdout
        sys.stdout = log

        Z = tt.rand(n, n.shape[0], 1)
        Y = rect_cross(
            func,
            Z,
            eps=eps,
            nswp=opts.get('nswp', 200),
            kickrank=opts.get('kickrank', 1),
            rf=opts.get('rf', 2),
            verbose=True
        )
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
