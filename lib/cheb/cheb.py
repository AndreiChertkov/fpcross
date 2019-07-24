import numpy as np

from .utils import polynomial, interpolate_1d

def prep(Y):
    '''
    Find coefficients A_i1...iN for interpolation of the N dimensional
    function by Chebyshev polynomials in the form
    f(x1, x2, ..., xN) =
    \sum_{i1, ..., iN} (a_i1...iN * T_i1(x1) * T_i2(x2) * ... * T_iN(xN)).

    INPUT:

    Y - tensor of function values on nodes of the Chebyshev mesh
    (for different axis numbers of points may be not equal)
    type: ndarray [N1, N2, ..., Ndim] of float

    OUTPUT:

    A - constructed tensor of coefficients
    type: ndarray [N1, N2, ..., Ndim] of float
    '''

    N = Y.shape
    d = len(N)
    A = Y.copy()

    for i in range(d):
        A = np.swapaxes(A, 0, i)
        A = A.reshape((N[i], -1))
        A = interpolate_1d(A)
        A = A.reshape(N)
        A = np.swapaxes(A, i, 0)

    return A

def calc(X, A, l):
    '''
    Calculate values of interpolated function in given x points.

    INPUT:

    X - values of x variable
    type: ndarray (or list) [dimensions, number of points] of float
    A - tensor of coefficients
    type: ndarray [dimensions] of float
    l - min-max values of variable for every dimension
    type: ndarray (or list) [dimensions, 2] of float

    OUTPUT:
    
    Y - approximated values of the function in given points
    type: ndarray [number of points] of float
    '''

    if not isinstance(X, np.ndarray): X = np.array(X)
    if not isinstance(l, np.ndarray): l = np.array(l)

    d = X.shape[0]
    n = X.shape[1]

    Y = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        B = A.copy()
        Z = (2. * X[:, j] - l[:, 1] - l[:, 0]) / (l[:, 1] - l[:, 0])
        for i in range(X.shape[0]):
            T = polynomial(A.shape[i], Z)
            B = np.tensordot(B, T[:,i], axes=([0], [0]))
        Y[j] = B

    return Y
