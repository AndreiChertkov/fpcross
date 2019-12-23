import numpy as np
from scipy.linalg import toeplitz


def difscheb(SG, m):
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


def dif1cheb(SG):
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

    return difscheb(SG, 1)[0]


def dif2cheb(SG):
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

    return difscheb(SG, 2)[1]
