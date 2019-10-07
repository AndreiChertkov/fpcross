import numpy as np
from scipy.linalg import toeplitz as _toeplitz

def dif_cheb(n, m, l_min=-1., l_max=+1.):
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
