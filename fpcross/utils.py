import numpy as np
from scipy.linalg import toeplitz


def dif_matrices(m, n, l_min, l_max):
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

    return D_list


def ode_solve(f, y0, t, n, h):
    y = y0.copy()
    for _ in range(1, n):
        k1 = h * f(y, t)
        k2 = h * f(y + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(y + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(y + k3, t + h)
        y += (k1 + 2 * (k2 + k3) + k4) / 6.
        t += h
    return y
