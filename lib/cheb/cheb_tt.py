import numpy as np

import tt

from .utils import polynomial_1d, interpolate_1d

def prep(Y, tol=1.0E-10, ni=0):
    '''
    Find coefficients A_i1...iN for interpolation of the N dimensional
    function by Chebyshev polynomials in the form
    f(x1, x2, ..., xN) =
    \sum_{i1, ..., iN} (a_i1...iN * T_i1(x1) * T_i2(x2) * ... * T_iN(xN))
    in the Tensor Train (TT) format.

    INPUT:

    Y - tensor of function values on nodes of the Chebyshev mesh
    (for different axis numbers of points may be not equal)
    type: TT tensor [N1, N2, ..., Ndim] of float
    tol - (optional) is the tolerance for the TT-approximation
    type: float
    ni  - (optional) is a number of dimensions that have to be skipped
    (are not transformed)
    type: int, <= dimensions

    OUTPUT:

    A - constructed tensor of coefficients
    type: tt.tensor [N1, N2, ..., Ndim] of float
    '''

    G = tt.tensor.to_list(Y)

    for i in range(ni, Y.d):
        sh = G[i].shape
        G[i] = np.swapaxes(G[i], 0, 1)
        G[i] = G[i].reshape((sh[1], -1))
        G[i] = interpolate_1d(G[i])
        G[i] = G[i].reshape((sh[1], sh[0], sh[2]))
        G[i] = np.swapaxes(G[i], 0, 1)

    A = tt.tensor.from_list(G)
    A = A.round(tol)

    return A

def calc(X, A, l, ni=0):
    '''
    Calculate values of interpolated function in given x points
    in the TT format.

    INPUT:

    X - values of x variable
    type: ndarray (or list) [dimensions, number of points] of float
    A - tensor of coefficients
    type: tt.tensor [dimensions] of float
    l - min-max values of variable for every dimension
    type: ndarray (or list) [dimensions, 2] of float
    ni  - (optional) is a number of dimensions that have to be skipped
    (are not transformed)
    type: int, <= dimensions

    OUTPUT:
    
    Y - approximated values of the function in given points
    type: ndarray [number of points] of float
    '''

    if not isinstance(X, np.ndarray): X = np.array(X)
    if not isinstance(l, np.ndarray): l = np.array(l)

    G = tt.tensor.to_list(A)
    Y = np.zeros(X.shape[1])

    for j in range(X.shape[1]):
        i = 0

        if i<=ni-1:
            Q = G[i][:, int(X[i, j]), :]
        else:
            Z = (2.*X[i, j] - l[i, 1] - l[i, 0]) / (l[i, 1] - l[i, 0])
            T = polynomial_1d(G[i].shape[1], Z)
            Q = np.tensordot(G[i], T, axes=([1], [0]))

        for i in range(1, X.shape[0]):
            if i <= ni-1:
                Q = np.dot(Q, G[i][:, int(X[i, j]), :])
                continue

            Z = (2.*X[i, j] - l[i, 1] - l[i, 0]) / (l[i, 1] - l[i, 0])
            T = polynomial_1d(G[i].shape[1], Z)
            Q = np.dot(Q, np.tensordot(G[i], T, axes=([1], [0])))

        Y[j] = Q[0, 0]

    return Y
