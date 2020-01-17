import numpy as np
from scipy.linalg import toeplitz


def poly(X, m, l=None):
    '''
    Calculate Chebyshev polynomials of order 0, 1, ..., (m-1)
    in all given X points.

    INPUT:

    X - values of x variable
    type1: float
    type2: list/ndarray [number of points] of float
    type3: list/ndarray [dimensions, number of points] of float
    * In case of only one point in 1D, it may be float (type1).
    * In case of 1D, it may be 1D ndarray or list (type2).

    m - max order of polynomial (function will calculate for 0, 1, ..., m-1)
    type: int >= 1

    l - min-max values of variable for each dimension
    type: list/ndarray [dimensions, 2] of float
    * Default limits are [-1, 1]. If l is given, then X values
    * will be scaled from l-limits to [-1, 1] limits.

    OUTPUT:

    T - Chebyshev polynomials of order 0, 1, ..., m-1 in given points
    type: ndarray [m, *X.shape] of float
    * If X is float, it will be ndarray [m].

    TODO Maybe replace input l by the corresponding grid.

    TODO Replace by more compact code (if len(X.shape) may be combined).
    '''

    m = int(m)
    if m <= 0:
        raise ValueError('Invalid parameter m (should be > 0).')

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
            l2 = np.repeat(l[0, 1], X.shape[0], axis=0)
        else:
            l1 = l[0, 0]
            l2 = l[0, 1]

        X = (2. * X - l2 - l1) / (l2 - l1)

    T = np.ones([m] + list(X.shape)) if len(X.shape) else np.ones([m, 1])
    if m >= 2:
        T[1, ] = X.copy()
        for k in range(2, m):
            T[k, ] = 2. * X * T[k - 1, ] - T[k - 2, ]

    return T if len(X.shape) else T.reshape(-1)


def difs(SG, m):
    '''
    Construct differentiation matrices on the Chebyshev grid
    of order 1, 2, ..., m and size n x n on the custom interval (l1, l2).

    INPUT:

    SG - spatial grid for construction of the matrices
    type: fpcross.Grid
    * Average limits (l1, l2)
    * and average number of the grid points (n0) will be used.

    m - maximum matrix order (will construct for 1, 2, ..., m)
    type: int >= 1

    OUTPUT:

    D - list of differentiation matrices (D1, D2, ..., Dm)
    type: tuple [m] of ndarray [n, n] of float

    TODO Check and update the code.
    '''

    m = int(m)
    if m <= 0:
        raise ValueError('Invalid parameter m (should be > 0).')

    if SG.k != 'c':
        raise ValueError('Invalid spatial grid (should be Chebyshev).')

    n = SG.n0
    l_min = SG.l1
    l_max = SG.l2

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

    C = toeplitz((-1.)**k)
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


def dif1(SG):
    '''
    Construct the 1th order differentiation matrix on the Chebyshev grid
    of size n x n on the custom interval (l1, l2).

    INPUT:

    SG - spatial grid for matrix construction
    type: fpcross.Grid
    * Average limits (l1, l2)
    * and average number of the grid points (n0) will be used.

    OUTPUT:

    D - the 1th order differentiation matrix
    type: ndarray [n, n] of float
    '''

    return difs(SG, 1)[0]


def dif2(SG):
    '''
    Construct the 2th order differentiation matrix on the Chebyshev grid
    of size n x n on the custom interval (l1, l2).

    INPUT:

    SG - spatial grid for matrix construction
    type: fpcross.Grid
    * Average limits (l1, l2)
    * and average number of the grid points (n0) will be used.

    OUTPUT:

    D - the 2th order differentiation matrix
    type: ndarray [n, n] of float
    '''

    return difs(SG, 2)[1]
