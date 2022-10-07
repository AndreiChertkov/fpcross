import numpy as np
import teneva
from time import perf_counter as tpc


from .fpcross import FPCross
from .utils import _renormalize_new


class FPCrossOUPDiag(FPCross):
    def _apply_exact_oup(self, G, X, rhs, t, cf=1, mu=0.):
        X0 = X
        X0, X1 = np.meshgrid(X0, X, indexing='xy')
        c = (1 - np.exp(-2. * rhs * t)) / rhs
        M = -((mu - X1 + (X0 - mu)*np.exp(-rhs*t))**2) / c
        M = cf*np.exp(M) / np.sqrt(np.pi * c)
        return M @ G

    def _step(self):
        tc = tpc()
        mu, a, b, n = self.eq.coef_mu, self.eq.a, self.eq.b, self.eq.n
        args = zip(self.Y0, self.X, np.diag(self.eq.coef_rhs), a, b, n)
        self.Y = [self._apply_exact_oup(G, X, rhs, self.t, (b-a)/n)
            for k, (G, X, rhs, a, b, n) in enumerate(args)]

        self.tc += tpc() - tc

        self._step_hist()

    def _step_init(self):
        tc = tpc()
        self.X = [np.linspace(a, b, n)
            for a, b, n in zip(self.eq.a, self.eq.b, self.eq.n)]
        self.Y0 = self.eq.build_r0()
        self.Y0 = _renormalize_new(self.Y0, self.eq.a, self.eq.b, self.eq.n)
        self.tc += tpc() - tc
