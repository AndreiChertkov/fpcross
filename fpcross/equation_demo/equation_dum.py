import numpy as np
import teneva


from ..equation import Equation


class EquationDum(Equation):
    """Dumbbell model, a special case of the FPE (it works only for d=3)."""

    def __init__(self, d=3, e=1.E-5, is_full=False, name='Dum'):
        super().__init__(d, e, is_full, name)

        if is_full:
            raise ValueError('This equation works only for TT-format')

        self.set_grid(n=60, a=-10., b=+10.)
        self.set_grid_time(m=100, t=10.)

        self.eta_list = []
        self.psi_list = []

        self.eta_real = +1.0328125
        self.psi_real = +2.071143

    def calc_eta(self, A):
        def func(X):
            R = teneva.cheb_get(X, A, self.a, self.b)
            return self.deta(X, R)

        Y_ = self.build(func)
        A_ = teneva.cheb_int(Y_)

        eta = teneva.cheb_sum(A_, self.a, self.b)
        self.eta_list.append(eta)

    def calc_psi(self, A):
        def func(X):
            R = teneva.cheb_get(X, A, self.a, self.b)
            return self.dpsi(X, R)

        Y_ = self.build(func)
        A_ = teneva.cheb_int(Y_)

        psi = teneva.cheb_sum(A_, self.a, self.b)
        self.psi_list.append(psi)

    def callback(self, solver):
        if not solver.is_last:
            return

        self.calc_eta(solver.A)
        self.calc_psi(solver.A)

        e_eta = abs(self.eta_real - self.eta_list[-1]) / abs(self.eta_real)
        e_psi = abs(self.psi_real - self.psi_list[-1]) / abs(self.psi_real)

        return f' | e_eta={e_eta:-8.2e} | e_psi={e_psi:-8.2e}'

    def dphi(self, X):
        X = X.T
        res = np.exp(-np.sum(X*X, axis=0) / 2. / self.p**2)
        res = 1. - self.a_opt / self.p**5 * res
        res = res * X
        return res.T

    def deta(self, X, r):
        p = self.dphi(X)
        return r * X[:, 0] * p[:, 1]  / self.b_opt

    def dpsi(self, X, r):
        p = self.dphi(X)
        return r * (X[:, 0] * p[:, 0] - X[:, 1] * p[:, 1]) / self.b_opt**2

    def f(self, X, t):
        X = X.T
        q = self.a_opt / 2. / self.p**5
        q *= np.exp(-np.sum(X*X, axis=0) / 2. / self.p**2)
        res = (q - 0.5) * X + self.K @ X
        return res.T

    def f1(self, X, t):
        X = X.T
        q = self.a_opt / 2. / self.p**5
        q *= np.exp(-np.sum(X*X, axis=0) / 2. / self.p**2)
        res = -0.5 + q - q / self.p**2 * X**2
        return res.T

    def init(self):
        self.with_rs = False
        self.with_rt = False

        self.a_opt = 0.1
        self.b_opt = 1.
        self.p = 0.5

        self.K = np.zeros((self.d, self.d))
        self.K[0, 1] = 1.
        self.K *= self.b_opt
