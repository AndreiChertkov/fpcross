import numpy as np
from scipy.linalg import solve_lyapunov


from ..equation import Equation


class EquationOUP(Equation):
    """Ornstein-Uhlenbeck process, a special case of the FPE."""

    def __init__(self, d=3, e=1.E-4, is_full=False, name='OUP'):
        super().__init__(d, e, is_full, name)

        self.set_coef_rhs()
        self.set_grid(n=30, a=-5., b=+5.)
        self.set_grid_time(m=100, t=5.)

    def f(self, X, t):
        return -X @ self.coef_rhs.T

    def f1(self, X, t):
        return -np.ones(X.shape) @ self.coef_rhs.T

    def init(self):
        self.with_rs = True
        self.with_rt = True if self.d == 1 else False

        self.W = solve_lyapunov(self.coef_rhs, 2. * self.coef * np.eye(self.d))
        self.Wi = np.linalg.inv(self.W)
        self.Wd = np.linalg.det(self.W)

    def rs(self, X):
        r = np.exp(-0.5 * np.diag(X @ self.Wi @ X.T))
        r /= np.sqrt(2**self.d * np.pi**self.d * self.Wd)
        return r

    def rt(self, X, t):
        if self.d != 1:
            raise ValueError('It is valid only for 1D case')

        K = self.coef_rhs[0, 0]
        S = (1. - np.exp(-2. * K * t)) / 2. / K
        S += self.coef * np.exp(-2. * K * t)
        a = 2. * S
        r = np.exp(-X * X / a) / np.sqrt(np.pi * a)
        return r.reshape(-1)

    def set_coef_rhs(self, coef_rhs=None):
        """Set coefficient matrix for the rhs.

        Args:
            coef_rhs (np.ndarray): coefficient matrix for the rhs (rhs = -1 *
                coef_rhs * x). It is np.ndarray of the shape [d, d].

        """
        if coef_rhs is None:
            self.coef_rhs = np.eye(self.d)
        else:
            self.coef_rhs = coef_rhs
