import numpy as np

def point(i, n, l):
    '''
    Get point of Chebyshev multidimensional (dim) grid for given multi index i.
    Points for every axis k are calculated as x = cos(i[k]*pi/(n[k]-1)),
    where n[k] is a total number of points for selected axis k, and then
    these points are scaled according to given limits l.

    INPUT:

    i - multi index for the grid point
    type: ndarray (or list) [dimensions] of int

    n - total number of points for each dimension
    type: ndarray (or list) [dimensions] of int

    l - min-max values of variable for each dimension
    type: ndarray (or list) [dimensions, 2] of float

    OUTPUT:

    x - grid point
    type: ndarray [dimensions] of float
    '''

    if not isinstance(i, np.ndarray): i = np.array(i)
    if not isinstance(n, np.ndarray): n = np.array(n)
    if not isinstance(l, np.ndarray): l = np.array(l)

    t = np.cos(np.pi * i / (n - 1))
    x = t * (l[:, 1] - l[:, 0]) / 2. + (l[:, 1] + l[:, 0]) / 2.
    return x

def interpolate_1d(Y):
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

    n_poi = Y.shape[0]

    Yext = np.zeros((2*n_poi-2, Y.shape[1]))
    Yext[0:n_poi, :] = Y[:, :]

    for k in range(n_poi, 2*n_poi-2):
        Yext[k, :] = Y[2*n_poi-k-2, :]

    A = np.zeros(Y.shape)
    for n in range(Y.shape[1]):
        A[:, n] = (np.fft.fft(Yext[:, n]).real/(n_poi - 1))[0:n_poi]
        A[0, n] /= 2.

    return A

def polynomial_1d(m, x):
    '''
    Calculate Chebyshev polynomials of order until m
    in given point.

    INPUT:

    m - upper bound for order of polynomial (will calculate for 0, 1, ..., m-1)
    type: int
    x - value of x variable
    type: float

    OUTPUT:

    T - Chebyshev polynomials of order 0, 1, ..., m-1
    type: ndarray [m] of float
    '''

    T = np.ones(m)
    if m == 1: return T

    T[1] = x
    for n in range(2, m):
        T[n] = 2. * x * T[n-1] - T[n-2]

    return T

def polynomial(m, X):
    '''
    Calculate Chebyshev polynomials of order until m
    in given points.

    INPUT:

    m - upper bound for order of polynomial (will calculate for 0, 1, ..., m-1)
    type: int
    X - values of x variable
    type: ndarray (or list) [dimensions, number of points] of float

    OUTPUT:

    T - Chebyshev polynomials
    type: ndarray [m, *X.shape] of float
    '''

    if not isinstance(X, np.ndarray): X = np.array(X)

    T = np.ones([m] + list(X.shape))
    if m == 1: return T

    T[1, :] = X.copy()
    for n in range(2, m):
        T[n, ] = 2. * X * T[n-1, ] - T[n-2, ]

    return T

def polynomial_der(m, X):
    '''
    Calculate Chebyshev polynomials' derivatives of order until m
    in given points.

    INPUT:

    m - upper bound for order of polynomial (will calculate for 0, 1, ..., m-1)
    type: int
    X - values of x variable
    type: ndarray (or list) [dimensions, number of points] of float

    OUTPUT:

    T - derivatives of Chebyshev polynomials
    type: ndarray [m, X.shape] of float
    '''

    if not isinstance(X, np.ndarray): X = np.array(X)

    T = np.zeros([m] + list(X.shape))
    if m == 1: return T

    Z = np.arccos(X)
    for n in range(1, m):
        T[n, ] = n * np.sin(n * Z) / np.sin(Z)

    return T
