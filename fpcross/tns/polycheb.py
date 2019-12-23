import numpy as np


def polycheb(X, m, l=None):
    '''
    Calculate Chebyshev polynomials of order 0, 1, ..., (m-1)
    in all given X points.

    INPUT:

    X - values of x variable
    type1: float
    type2: list [number of points]
    type3: ndarray [number of points]
    type4: list [dimensions, number of points] of float
    type5: ndarray [dimensions, number of points] of float
    * In case of only one point in 1D, it may be float (type1).
    * In case of 1D, it may be 1D ndarray (or list) (type2 or type3).

    m - max order of polynomial (function will calculate for 0, 1, ..., m-1)
    type: int >= 1

    l - min-max values of variable for each dimension
    type1: list [dimensions, 2] of float
    type2: ndarray [dimensions, 2] of float
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
