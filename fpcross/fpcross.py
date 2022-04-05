import numpy as np
from scipy.linalg import expm
from scipy.linalg import toeplitz
import teneva
from time import perf_counter as tpc
from tqdm import tqdm


from .plot import plot
from .plot import plot_spec


class FPCross:
    def __init__(self, eq, with_hist=False):
        """Class that represents the solver for the Fokker-Planck equation.

        Args:
            eq (Equation): equation class instance.
            with_hist (bool): if flag is set, then accuracy of the result will
                be checked after each time step. Otherwise, the accuracy check
                will be performed only after the solver has finished running.

        """
        self.eq = eq
        self.is_full = self.eq.is_full
        self.with_hist = bool(with_hist)

        self.A = None  # Interpolation coefficients for the Y
        self.W = None  # Copy of the conv. solution from the previous time step
        self.Y = None  # Current solution r(x, t); it is tensor on the grid
        self.Z = None  # Matrix exponent for finite difference matrices
        self.s = None  # Current value of Y integral over the spatial domain

        self.m = 0         # Current time-step
        self.t = 0.        # Current value of the time variable
        self.t_calc = 0.   # Computation time (duration)

        # History for ranks and errors while computation proccess
        self.es_list = []
        self.et_list = []
        self.r_list = []
        self.s_list = []

    def plot(self, fpath=None, is_spec=False):
        """Plot the computation result.

        Args:
            fpath (str): optional path to png-file to save the figure.
            is_spec (bool): this flag should be set for the "EquationDum".

        """

        if is_spec:
            plot_spec(self, fpath)
        else:
            plot(self, fpath)

    def solve(self):
        """Solve the Fokker-Planck equation."""
        self._step_init()

        for m in range(1, self.eq.m + 1):
            self.m = m
            self.t = self.eq.h * self.m

            self._step()
            self._step_proc()
            self.eq.callback(self)

        self._step_post()

    def _check_rs(self):
        if not self.eq.with_rs:
            return

        if self.eq.Ys is None:
            self.eq.build_rs()

        e = teneva.accuracy(self.Y, self.eq.Ys)
        self.es_list.append(e)

        self.text += f'e_s={e:-8.2e}|'
        return e

    def _check_rt(self):
        if not self.eq.with_rt:
            return

        self.eq.build_rt(self.t)

        e = teneva.accuracy(self.Y, self.eq.Yt)
        self.et_list.append(e)

        self.text += f'e_t={e:-8.2e}|'
        return e

    def _diff_init(self):
        D1, D2 = dif_matrices(2, self.eq.n[0], self.eq.a[0], self.eq.b[0])
        h0 = self.eq.h / 2
        J0 = np.eye(self.eq.n[0])
        J0[+0, +0] = 0.
        J0[-1, -1] = 0.
        self.Z = expm(h0 * self.eq.coef * J0 @ D2)

        if self.is_full:
            Z0 = self.Z.copy()
            for k in range(1, self.eq.d):
                self.Z = np.kron(self.Z, Z0)

    def _interpolate(self):
        if self.is_full:
            self.A = teneva.cheb_int_full(self.Y)
        else:
            self.A = teneva.cheb_int(self.Y)
            self.A = teneva.truncate(self.A, self.eq.e)

    def _renormalize(self):
        self._interpolate()

        if self.is_full:
            self.s = teneva.cheb_sum_full(self.A, self.eq.a, self.eq.b)
            self.Y = (1. / self.s) * self.Y
        else:
            self.s = teneva.cheb_sum(self.A, self.eq.a, self.eq.b)
            self.Y = teneva.mul(1. / self.s, self.Y)

    def _step(self):
        t_ = tpc()
        self._step_diff()
        self._interpolate()
        self._step_conv()
        self._renormalize()
        self.W = teneva.copy(self.Y)
        self._step_diff()
        self.t_calc += tpc() - t_

    def _step_conv(self):
        def func(y, t):
            X, r = y[:, :-1], y[:, -1]
            f0 = self.eq.f(X, self.t)
            f1 = self.eq.f1(X, self.t)
            f1_mult = - np.sum(f1, axis=1) * r
            f1_mult = f1_mult.reshape(-1, 1)
            return np.hstack([f0, f1_mult])

        def step_conv(X):
            X0 = ode_solve(self.eq.f, X, self.t, 2, -self.eq.h)

            if self.is_full:
                w0 = teneva.cheb_get_full(X0, self.A, self.eq.a, self.eq.b)
            else:
                w0 = teneva.cheb_get(X0, self.A, self.eq.a, self.eq.b)

            y0 = np.hstack([X0, w0.reshape(-1, 1)])

            y1 = ode_solve(func, y0, self.t-self.eq.h, 2, self.eq.h)

            w1 = y1[:, -1]

            return w1

        self.Y = self.eq.build(step_conv, self.W)

    def _step_diff(self):
        if self.is_full:
            self.Y = self.Y.reshape(-1, order='F')
            self.Y = self.Z @ self.Y
            self.Y = self.Y.reshape(self.eq.n, order='F')
        else:
            for k in range(self.eq.d):
                self.Y[k] = np.einsum('ij,kjm->kim', self.Z, self.Y[k])
            self.Y = teneva.truncate(self.Y, self.eq.e)

    def _step_init(self):
        t_ = tpc()
        self._diff_init()
        self.Y = self.eq.build_r0()
        self.W = teneva.copy(self.Y)
        self.t_calc += tpc() - t_
        self.tqdm = tqdm(desc='Solve', unit='step', total=self.eq.m-1, ncols=90)

    def _step_post(self):
        self.tqdm.close()

    def _step_proc(self):
        self.text = f'|t={self.t:-4.2f}|'

        if not self.is_full:
            r = teneva.erank(self.Y)
            self.r_list.append(r)
            self.text += f'r={r:-3.1f}|'

        if self.with_hist or self.m == self.eq.m:
            self._check_rs()
            self._check_rt()

        self.s_list.append(self.s)

        self.tqdm.set_postfix_str(self.text, refresh=True)
        self.tqdm.update(1)


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
