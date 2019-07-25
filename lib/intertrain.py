import sys
import time

import numpy as np
from scipy.linalg import toeplitz

import tt
from tt.cross.rectcross import cross as rect_cross

class SparseArray:
    '''
    Keep elements of (sparse) multidimensional array as elements of dictionary.
    '''

    def __init__(self):
        self.elements = {}

    def ind_to_tuple(self, ind):
        if isinstance(ind,(tuple)):
            return ind
        else:
            return tuple(ind)

    def read(self, ind):
        try:
            value = self.elements[self.ind_to_tuple(ind)]
        except KeyError:
            value = None
        return value

    def add(self, ind, value):
        self.elements[self.ind_to_tuple(ind)] = value

    def add_if_new(self, ind, value, eps=1.E-16):
        value0 = self.read(ind)
        if value0==None:
            self.add(ind, value)
            return
        if abs(value-value0) > eps:
            self.add(ind, value)

class Intertrain(object):
    '''
    Class for multidimensional function interpolation
    by Chebyshev polynomials in the TT-format with cross approximation.
    '''

    def __init__(self, n, l, ns=0, eps=1.E-6):
        '''
        INPUT:

        n - total number of points for each dimension
        type: ndarray (or list) [dimensions] of int

        l - min-max values of variable for each dimension
        type: ndarray (or list) [dimensions, 2] of float

        ns - (optional) is a number of dimensions that should be skipped
        (are not transformed)
        type: int, <= dimensions

        eps - (optional) is the desired accuracy of the approximation
        type: float
        '''

        self.n = n if isinstance(n, np.ndarray) else np.array(n)
        self.l = l if isinstance(l, np.ndarray) else np.array(l)
        self.d = self.l.shape[0]
        self.ns = ns
        self.eps = eps

        self.init()

    def init(self, f=None, Y=None):
        '''
        Init main calculation parameters and training data, using
        given function f (cross approximation will be applied in this case)
        or given tt-tensor Y with data.

        INPUT:

        f - (optional) function that calculate tensor element for given point
        (if ns>0, then first ns dimensions should be simple integer indices)
        type: function
            input type: ndarray [dimensions] of float
            output type: float

        Y - (optional) tensor of function values on nodes of the Chebyshev mesh
        (for different axes different numbers of points may be used)
        type: TT-tensor [n_1, n_2, ..., n_dim] of float
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

        self._t_init = time.time()

        self.f = f
        self.Y = self.cross(self.f) if self.f else Y
        self.A = None

        self._t_init = time.time() - self._t_init

    def prep(self):
        '''
        Build tensor of interpolation coefficients according to training data.
        '''

        if self.Y is None:
            raise ValueError('Train data is not set. Call "init" before.')

        self._t_prep = time.time()

        G = tt.tensor.to_list(self.Y)

        for i in range(self.ns, self.Y.d):
            sh = G[i].shape
            G[i] = np.swapaxes(G[i], 0, 1)
            G[i] = G[i].reshape((sh[1], -1))
            G[i] = self.interpolate(G[i])
            G[i] = G[i].reshape((sh[1], sh[0], sh[2]))
            G[i] = np.swapaxes(G[i], 0, 1)

        self.A = tt.tensor.from_list(G)
        self.A = self.A.round(self.eps)

        self._t_prep = time.time() - self._t_prep

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

        self._t_calc = time.time()

        if not isinstance(X, np.ndarray): X = np.array(X)

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

        self._t_calc = (time.time() - self._t_calc) / X.shape[1]

        return Y

    def dif2(self):
        '''
        Calculate second order differentiation matrix on Chebyshev grid
        on interval [-1,1].

        OUTPUT:

        D - second order differentiation matrix
        type: ndarray [num. poi., num. poi.] of float

        TODO:

        Add (check) support for custom limits.
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

        f - (optional) function that calculate tensor element for given point
        (if not set, then function from init arg will be used)
        (if ns>0, then first ns dimensions should be simple integer indices)
        type: function
            input type: ndarray [dimensions] of float
            output type: float

        npoi - (optional) number of points for error check
        type: int

        a - (optional) minimum value for random check point
        type: float

        b - (optional) maximum value for random check point
        type: float
        '''

        print('------------------ Parameters')
        print('Dimensions       : %8d'%self.d)
        print('Accuracy         : %8.2e'%self.eps)
        for i, [n, l] in enumerate(zip(self.n, self.l)):
            opts = (i+1, n, l[0], l[1])
            print('Dim %-2d | Poi %-3d | Min %-6.3f | Max %-6.3f |'%opts)

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

        f - (optional) function that calculate tensor element for given point
        (if not set, then function from init arg will be used)
        (if ns>0, then first ns dimensions should be simple integer indices)
        type: function
            input type: ndarray [dimensions] of float
            output type: float

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

    def point(self, i):
        '''
        Get point of Chebyshev multidimensional (dim) grid
        for given multi index i. Points for every axis k are calculated as
        x = cos(i[k]*pi/(n[k]-1)), where n[k] is a total number of points
        for selected axis k, and then these points are scaled according to
        given limits l.

        INPUT:

        i - multi index for the grid point
        type: ndarray (or list) [dimensions] of int

        OUTPUT:

        x - grid point
        type: ndarray [dimensions] of float
        '''

        if not isinstance(i, np.ndarray): i = np.array(i)

        t = np.cos(np.pi * i / (self.n - 1))
        x = t * (self.l[:, 1] - self.l[:, 0]) / 2.
        x+= (self.l[:, 1] + self.l[:, 0]) / 2.
        return x

    def points_1d(self, n=None, l1=None, l2=None):
        '''
        Get points of 1D Chebyshev grid (from max to min value).
        Limits and number of points for the first spatial axis are used
        if related optional parameters are not set.

        OUTPUT:

        x - grid points
        type: ndarray [number of points] of float
        '''


        if n is None: n = self.n[0]
        if l1 is None: l1 = self.l[0, 0]
        if l2 is None: l2 = self.l[0, 1]

        i = np.arange(n)
        t = np.cos(np.pi * i / (n - 1))
        x = t * (l2-l1) / 2. + (l1+l2) / 2.

        return x

    def points_2d(self):
        '''
        Get points of 2D Chebyshev grid (from max to min value).
        Limits and number of points for the 1th and 2th spatial axis are used
        if related optional parameters are not set.

        OUTPUT:

        x - grid points
        type: ndarray [2, number of points] of float
        '''

        x1 = self.points_1d(self.n[0], self.l[0, 0], self.l[0, 1])
        x2 = self.points_1d(self.n[1], self.l[1, 0], self.l[1, 1])
        x1, x2 = np.meshgrid(x1, x2)
        x = np.array([x1, x2]).reshape((2, -1))

        return x

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

    def polynomial_der(self, X, m):
        '''
        Calculate Chebyshev polynomials' derivatives of order until m
        in given points.

        INPUT:

        X - values of x variable
        type: ndarray (or list) [dimensions, number of points] of float

        m - upper bound for order of polynomial (will calculate for 0, ..., m-1)
        type: int

        OUTPUT:

        T - derivatives of Chebyshev polynomials of order 0, 1, ..., m-1
        type: ndarray [m, X.shape] of float
        '''

        if not isinstance(X, np.ndarray): X = np.array(X)

        T = np.zeros([m] + list(X.shape))
        if m == 1: return T

        Z = np.arccos(X)
        for n in range(1, m):
            T[n, ] = n * np.sin(n * Z) / np.sin(Z)

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
        x/= (self.l[n, 1] - self.l[n, 0])

        T = np.ones(m)
        if m == 1: return T

        T[1] = x
        for n in range(2, m):
            T[n] = 2. * x * T[n-1] - T[n-2]

        return T

    def interpolate(self, Y):
        '''
        Find coefficients A_i for interpolation of 1D functions by Chebyshev
        polynomials f(x) = \sum_{i} (A_i * T_i(x)) using fast Fourier transform.
        It can find coefficients for several functions on one call
        if all functions have equal numbers of grid points.

        INPUT:

        Y - values of function at the mesh nodes
        type: ndarray (or list) [number of points, number of functions] of float

        OUTPUT:

        A - constructed matrix of coefficients
        type: ndarray [number of points, number of functions] of float
        '''

        if not isinstance(Y, np.ndarray): Y = np.array(Y)

        n = Y.shape[0]

        Yext = np.zeros((2*n - 2, Y.shape[1]))
        Yext[0:n, :] = Y[:, :]

        for k in range(n, 2*n - 2):
            Yext[k, :] = Y[2*n - k - 2, :]

        A = np.zeros(Y.shape)
        for k in range(Y.shape[1]):
            A[:, k] = (np.fft.fft(Yext[:, k]).real/(n - 1))[0:n]
            A[0, k] /= 2.

        return A

    def cross(self, f):
        '''
        Calculate function's values tensor on the multidimensional
        Chebyshev mesh (for dimensions that are not skipped)
        by cross approximation method in the tensor train (TT) format.

        INPUT:

        f - (optional) function that calculate tensor element for given point
        (if ns>0, then first ns dimensions should be simple integer indices)
        type: function
            input type: ndarray [dimensions] of float
            output type: float

        OUTPUT:

        Y - approximated tensor of the function's values in given points
        type: tt.tensor [dimensions] of float
        '''

        def calc_vals(Ind):
            '''
            Calculate tensor's elements for given indices.

            INPUT:

            Ind - are the indices of tensor
            type: ndarray [number of points, dimensions] of int

            OUTPUT:

            Y - elements of the tensor
            type: ndarray [number of points] of float
            '''

            Y = np.zeros(Ind.shape[0])
            for i in range(Ind.shape[0]):
                Ycurr = Ytmp.read(Ind[i, :])
                if Ycurr is not None:
                    Y[i] = Ycurr
                    continue
                x = self.point(Ind[i, :])
                # x = np.hstack((Ind[i, :n_dim_ind], x))
                Ycurr = self.f(x)
                Ytmp.add(Ind[i, :], Ycurr)
                self.crs_res['evals'] += 1
            return Y

        file_log = './tmp.txt'

        Ytmp = SparseArray()

        try:
            log = open(file_log, 'w')
        except:
            pass
        else:
            stdout0 = sys.stdout
            sys.stdout = log

        Y0 = tt.rand(self.n, self.n.shape[0], 1)
        Y = rect_cross(
            calc_vals,
            Y0,
            eps=self.eps,
            nswp=200,
            kickrank=1,
            rf=2,
            verbose=True
        )

        try:
            log.close()
            sys.stdout = stdout0
        except:
            pass

        try:
            log = open(file_log, 'r')
            res = log.readlines()[-1].split('swp: ')[1]
            self.crs_res['iters'] = int(res.split('/')[0])+1
            res = res.split('er_rel = ')[1]
            self.crs_res['err_rel'] = float(res.split('er_abs = ')[0])
            res = res.split('er_abs = ')[1]
            self.crs_res['err_abs'] = float(res.split('erank = ')[0])
            res = res.split('erank = ')[1]
            self.crs_res['erank'] = float(res.split('fun_eval')[0])
        except:
            pass

        return Y
