import numpy as np
import teneva
import time


from ..equation import Equation


class EquationDum(Equation):
    def __init__(self, d=3, e=1.E-5, is_full=False, name='Dum'):
        """Dumbbell model, a special case of the FPE (it works only for d=3)."""
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
            R = teneca.cheb_get(X, A, self.a, self.b)
            return self.deta(X, R)

        Y_ = self.build(func)
        A_ = teneva.cheb_int(Y_)

        eta = teneva.cheb_sum(A_, self.a, self.b)
        self.eta_list.append(eta)

    def calc_psi(self, A):
        def func(X):
            R = teneca.cheb_get(X, A, self.a, self.b)
            return self.dpsi(X, R)

        Y_ = self.build(func)
        A_ = teneva.cheb_int(Y_)

        psi = teneva.cheb_sum(A_, self.a, self.b)
        self.psi_list.append(psi)

    def callback(self, solver):
        A = teneva.cheb_int(solver.Y)
        self.calc_eta(A)
        self.calc_psi(A)

        if True:
            time.sleep(0.3)
            print('\n')
            e_eta = abs(self.eta_real - self.eta_list[-1]) / abs(self.eta_real)
            e_psi = abs(self.psi_real - self.psi_list[-1]) / abs(self.psi_real)
            print(f'Eta err : {e_eta:-8.2e} | Psi err : {e_psi:-8.2e}')
            print('\n');
            time.sleep(0.3)

    def dphi(self, X):
        res = np.exp(-np.sum(X*X, axis=0) / 2. / self.p**2)
        res = 1. - self.a_opt / self.p**5 * res
        res = res * X
        return res

    def deta(self, X, r):
        p = self.dphi(X)
        return r * X[0, :] * p[1, :]  / self.b_opt

    def dpsi(self, X, r):
        p = self.dphi(X)
        return r * (X[0, :] * p[0, :] - X[1, :] * p[1, :]) / self.b_opt**2

    def f(self, X, t):
        q = self.a_opt / 2. / self.p**5
        q *= np.exp(-np.sum(X*X, axis=1) / 2. / self.p**2)
        return (q - 0.5) * X + self.K @ X

    def f1(self, X, t):
        q = self.a_opt / 2. / self.p**5
        q *= np.exp(-np.sum(X*X, axis=1) / 2. / self.p**2)
        return -0.5 + q - q / self.p**2 * X**2

    def init(self):
        self.with_rs = False
        self.with_rt = False

        self.a_opt = 0.1
        self.b_opt = 1.
        self.p = 0.5

        self.K = np.zeros((self.d, self.d))
        self.K[0, 1] = 1.
        self.K *= self.b_opt
