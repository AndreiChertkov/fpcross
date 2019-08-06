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
    '''

    def __init__(self, n, l, ns=0, eps=1.E-6, with_tt=True):
        '''
        INPUT:

        n - total number of points for each dimension
        type: ndarray (or list) [dimensions] of int >= 2

        l - min-max values of variable for each dimension
        type: ndarray (or list) [dimensions, 2] of float

        ns - (optional) is a number of dimensions that should be skipped
        (are not transformed while interpolation)
        type: int 0 <= and <= dimensions

        eps - (optional) is the desired accuracy of the approximation
        (is used in tt-round and cross approximation operations)
        type: float > 0

        with_tt - (optional) flag; if true, then TT-format will be used
        instead of dense (numpy) format
        type: bool
        '''

        self.n = n if isinstance(n, np.ndarray) else np.array(n)
        self.l = l if isinstance(l, np.ndarray) else np.array(l)
        self.d = self.l.shape[0]
        self.ns = ns
        self.eps = eps
        self.with_tt = with_tt

        if self.n.shape[0] != self.l.shape[0]:
            raise IndexError('Shape mismatch for n and l params.')


        self.init()

    def copy(self):
        '''
        Create a copy of the class instance.

        OUTPUT:

        intertrain - new class instance
        type: Intertrain

        '''

        return Intertrain(self.n, self.l, self.ns, self.eps, self.with_tt)

    def init(self, f=None, Y=None, fpath='./tmp.txt'):
        '''
        Init main calculation parameters and training data, using
        given function f (cross approximation will be applied if TT-format
        is used) or given tensor Y (in numpy or TT-format) of data.

        INPUT:

        f - (optional) function that calculate tensor elements for given points
        (if ns>0, then first ns dimensions should be simple integer indices)
        type: function
            inp type: ndarray [dimensions, number of points] of float
            output type: ndarray [number of points] of float

        Y - (optional) tensor of function values on nodes of the Chebyshev mesh
        (for different axes different numbers of points may be used)
        type: ndarray or TT-tensor [n_1, n_2, ..., n_dim] of float

        fpath - (optional) file for output info
        type: str

        OUTPUT:

        self - class instance
        type: Intertrain
        '''

        self._t_init = 0. # Time of training data collection
        self._t_prep = 0. # Time of interpolation coefficients calculation
        self._t_calc = 0. # Average time of interpolation evaluation for 1 poi
        self._t_func = 0. # Average time of function evaluation for 1 poi

        self.err = 0.
        self.crs_res = {
            'evals': 0,
            'iters': 0,
            'erank': 0.,
            'err_rel': 0.,
            'err_abs': 0.,
        }

        self.f = f
        self.Y = Y
        self.A = None

        if self.f is None or self.Y is not None:
            return self

        self._t_init = time.time()

        if self.with_tt:
            def func(ind):
                self.crs_res['evals'] += ind.shape[0]
                self._t_func = time.time()
                Y = self.f(self.pois(ind.T))
                self._t_func = (time.time() - self._t_func) / ind.shape[0]
                return Y

            log = open(fpath, 'w')
            stdout0 = sys.stdout
            sys.stdout = log

            Y0 = tt.rand(self.n, self.n.shape[0], 1)
            self.Y = rect_cross(
                func,
                Y0,
                eps=self.eps,
                nswp=200,
                kickrank=1,
                rf=2,
                verbose=True
            )

            log.close()
            sys.stdout = stdout0

            log = open(fpath, 'r')
            res = log.readlines()[-1].split('swp: ')[1]
            self.crs_res['iters'] = int(res.split('/')[0])+1
            res = res.split('er_rel = ')[1]
            self.crs_res['err_rel'] = float(res.split('er_abs = ')[0])
            res = res.split('er_abs = ')[1]
            self.crs_res['err_abs'] = float(res.split('erank = ')[0])
            res = res.split('erank = ')[1]
            self.crs_res['erank'] = float(res.split('fun_eval')[0])

        else:
            self.Y = None
            raise NotImplementedError('Is not implemented.')

        self._t_init = time.time() - self._t_init

        return self

    def prep(self):
        '''
        Build tensor of interpolation coefficients according to training data.
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
                G[i] = interpolate(G[i])
                G[i] = G[i].reshape((sh[1], sh[0], sh[2]))
                G[i] = np.swapaxes(G[i], 0, 1)

            self.A = tt.tensor.from_list(G)
            self.A = self.A.round(self.eps)
        else:
            self.A = self.Y.copy()

            for i in range(len(self.Y.shape)):
                self.A = np.swapaxes(self.A, 0, i)
                self.A = self.A.reshape((self.Y.shape[i], -1))
                self.A = interpolate(self.A)
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
        '''

        if self.A is None:
            raise ValueError('Interpolation is not done. Call "prep" before.')

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self._t_calc = time.time()

        if self.with_tt:
            G = tt.tensor.to_list(self.A)
            Y = np.zeros(X.shape[1])

            for j in range(X.shape[1]):
                i = 0

                if i <= self.ns - 1:
                    Q = G[i][:, int(X[i, j]), :]
                else:
                    T = self.polynomial_trans(X[i, j], G[i].shape[1], i)
                    Q = np.tensordot(G[i], T, axes=([1], [0]))

                for i in range(1, X.shape[0]):
                    if i <= self.ns - 1:
                        Q = np.dot(Q, G[i][:, int(X[i, j]), :])
                        continue

                    T = self.polynomial_trans(X[i, j], G[i].shape[1], i)
                    Q = np.dot(Q, np.tensordot(G[i], T, axes=([1], [0])))

                Y[j] = Q[0, 0]
        else:
            Y = np.zeros(X.shape[1])

            for j in range(X.shape[1]):
                A_tmp = self.A.copy()
                T = (2.*X[:, j]-x_lim[:, 1]-x_lim[:, 0]) / (x_lim[:, 1]-x_lim[:, 0])

                for i in range(X.shape[0]):
                    Tch = ch_ut.polynomial(self.A.shape[i], T)
                    A_tmp = np.tensordot(A_tmp, Tch[:,i], axes=([0], [0]))
                Y[j] = A_tmp

        self._t_calc = (time.time() - self._t_calc) / X.shape[1]

        return Y

    def dif2(self):
        '''
        Calculate second order differentiation matrix on Chebyshev grid
        on interval [-1, 1].

        OUTPUT:

        D - second order differentiation matrix
        type: ndarray [num. poi., num. poi.] of float

        TODO: add (check) support for custom limits.
        '''

        m = self.n[0] - 1
        m1 = np.int(np.floor((m + 1) / 2))
        m2 = np.int(np.ceil((m + 1) / 2))
        k = np.arange(m+1)
        th = k * np.pi / m

        T = np.tile(th/2, (m + 1, 1))
        DX = 2*np.sin(T.T+T)*np.sin(T.T-T)
        DX[m1:, :] = -np.flipud(np.fliplr(DX[0:m2, :]))
        DX[range(m + 1), range(m + 1)] = 1.
        DX = DX.T

        Z = 1 / DX
        Z[range(m + 1), range(m + 1)] = 0.

        C = toeplitz((-1.)**k)
        C[+0, :] *= 2
        C[-1, :] *= 2
        C[:, +0] *= 0.5
        C[:, -1] *= 0.5

        D = np.eye(m + 1)
        for ell in range(2):
            D = (ell + 1) * Z * (C * np.tile(np.diag(D), (m + 1, 1)).T - D)
            D[range(m + 1), range(m + 1)] = -np.sum(D, axis=1)

        return D

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

        if self.with_tt:
            print('------------------ Result (cross appr)')
            print('Func. evaluations: %8d'%self.crs_res['evals'])
            print('Cross iterations : %8d'%self.crs_res['iters'])
            print('Av. tt-rank      : %8.2e'%self.crs_res['erank'])
            print('Cross err (rel)  : %8.2e'%self.crs_res['err_rel'])
            print('Cross err (abs)  : %8.2e'%self.crs_res['err_abs'])

        if self.A is not None and (f or self.f):
            self.check(f or self.f, npoi, a, b)

            print('------------------ Test (random points)')
            print('Number of points : %d'%npoi)
            print('Error (max)      : %8.2e '%np.max(self.err))
            print('Error (mean)     : %8.2e '%np.mean(self.err))
            print('Error (min)      : %8.2e '%np.min(self.err))

        print('------------------ Time')
        print('Init             : %8.2e sec. '%self._t_init)
        print('Prep             : %8.2e sec. '%self._t_prep)
        print('Calc (average)   : %8.2e sec. '%self._t_calc)
        print('Func (average)   : %8.2e sec. '%self._t_func)

        print('------------------')

    def check(self, f=None, npoi=10, a=-1., b=1.):
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
        '''

        f = f or self.f
        if f is None:
            raise ValueError('Interpolated function is not set.')

        X = a + np.random.random((self.d, npoi)) * (b - a)
        Y_real = f(X)
        Y_calc = self.calc(X)
        self.err = np.abs((Y_calc - Y_real) / Y_real)

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
        X += (l2 + l1) / 2.

        return X

    def grid(self):
        '''
        Get all points of flatten multidimensional Chebyshev grid
        (from max to min value).

        OUTPUT:

        X - grid points
        type: ndarray [dimensions, number of n1 * n2 * ... * nd] of float
        '''

        I = [np.arange(self.n[d]).reshape(1, -1) for d in range(self.d)]
        I = np.meshgrid(*I)
        I = np.array(I).reshape((self.d, -1))

        return self.pois(I)

    def polynomial(self, X, m):
        '''
        Calculate Chebyshev polynomials of order until m
        in given points.

        INPUT:

        X - values of x variable
        type: ndarray (or list) [dimensions, number of points] of float

        m - upper bound for order of polynomial (will calculate for 0, ..., m-1)
        type: int

        OUTPUT:

        T - Chebyshev polynomials of order 0, 1, ..., m-1
        type: ndarray [m, *X.shape] of float
        '''

        if not isinstance(X, np.ndarray): X = np.array(X)

        T = np.ones([m] + list(X.shape))
        if m == 1: return T

        T[1, :] = X.copy()
        for n in range(2, m):
            T[n, ] = 2. * X * T[n-1, ] - T[n-2, ]

        return T

    def polynomial_trans(self, x, m, n):
        '''
        Calculate Chebyshev polynomials of order until m
        in given point x (with scaling) related to n-th spatial axis.

        INPUT:

        x - value of x variable
        type: float

        m - upper bound for order of polynomial (will calculate for 0, ..., m-1)
        type: int

        n - number of spatial axis (0, ..., dim-1)
        type: int

        OUTPUT:

        T - Chebyshev polynomials of order 0, 1, ..., m-1
        type: ndarray [m] of float
        '''


        x = (2. * x - self.l[n, 1] - self.l[n, 0])
        x /= (self.l[n, 1] - self.l[n, 0])

        T = np.ones(m)
        if m == 1: return T

        T[1] = x
        for n in range(2, m):
            T[n] = 2. * x * T[n-1] - T[n-2]

        return T

def polynomials(X, m):
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

    OUTPUT:

    T - Chebyshev polynomials of order 0, 1, ..., m in given points
    type: ndarray [m+1, *X.shape] of float
    * if X is float, it will be ndarray [m+1]
    '''

    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if len(X.shape):
        T = np.ones([m+1] + list(X.shape))
    else:
        T = np.ones([m+1, 1])

    if m > 0:
        T[1,] = X.copy()
        for k in range(2, m+1):
            T[k, ] = 2. * X * T[k-1, ] - T[k-2, ]

    return T if len(X.shape) else T.reshape(-1)

def interpolate(Y):
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
    A = np.fft.fft(V, axis=0).real
    A = A[:n, :] / (n - 1)
    A[0, :] /= 2.
    A[n-1, :] /= 2.
    return A
