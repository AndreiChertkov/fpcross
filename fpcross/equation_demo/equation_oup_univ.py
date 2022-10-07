import numpy as np
import teneva


from .equation_oup import EquationOUP
from ..utils import _renormalize_new


class EquationOUPUniv(EquationOUP):
    def __init__(self, *args, **kwargs):
        self.set_coef_mu(kwargs.pop('mu', 0))

        super().__init__(*args, **kwargs)

    def build(self, func, Y0=None, e=None):
        self.cross_info = {}
        self.cross_cache = {} if self.cross_with_cache else None

        if Y0 is None:
            Y0 = teneva.tensor_rand(self.n, self.cross_r)

        e = e or self.e
        f = lambda I: func(teneva.ind_to_poi(I, self.a, self.b, self.n, 'uni'))
        Y = teneva.cross(f, Y0, self.cross_m, e, self.cross_nswp,
            self.cross_tau, self.cross_dr_min, self.cross_dr_max,
            self.cross_tau0, self.cross_k0, self.cross_info, self.cross_cache)

        return teneva.truncate(Y, e)

    def build_rs_(self):
        # TODO: This function is completely identical to the function from the
        # base class. Why we need it?

        if not self.with_rs:
            self.Ys = None
            return self.Ys

        func = self.rs
        self.cross_nswp = 100
        self.Ys = self.build(func, e=self.e/100)
        self.Ys = _renormalize_new(self.Ys, self.a, self.b, self.n)
        return self.Ys

    def rs_(self, X):
        # TODO: What does it mean?
        self.log_M = (X - self.coef_mu) @ self.Wi @ (X.T - self.coef_mu)
        self.log_M =  -0.5 * np.diag(self.log_M)
        r = np.exp(self.log_M)
        Det = np.sqrt(2**self.d * np.pi**self.d * self.Wd)
        return r / Det

    def set_coef_mu(self, mu=0.):
        self.coef_mu = mu
