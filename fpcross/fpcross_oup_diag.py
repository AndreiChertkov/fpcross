import numpy as np
import teneva
from time import perf_counter as tpc


from .fpcross import FPCross
from .utils import _renormalize_new


class FPCrossOUPDiag(FPCross):
    def _apply_exact_oup(self, c, X, a, mu, t, cf=1):
        X0 = X
        #if quad is 'trap': #Trapezoidal rule
        #    X0 = X*....

        X0, X1 = np.meshgrid(X0, X, indexing='xy')
        ae2a = (1 - np.exp(-2*a*t)) / a
        M = (-( mu - X1 + (X0 - mu)*np.exp(-a*t) )**2) / ae2a

        Det = np.sqrt(np.pi*ae2a)

        M = cf*np.exp(M)
        M /= Det

        return M @ c

    def _step(self):
        tc = tpc()
        mu, a, b, n = self.eq.coef_mu, self.eq.a, self.eq.b, self.eq.n
        args = zip(self.Y0, np.diag(self.eq.coef_rhs), self.X, a, b, n)
        self.Y = [self._apply_exact_oup(G, X, A_i, mu, self.t, (b-a)/n)
            for k, (G, A_i, X, a, b, n) in enumerate(args)]

        # self.Y = _renormalize_new(self.Y, self.eq.a, self.eq.b, self.eq.n)

        self.tc += tpc() - tc

        self._step_hist()

    def _step_init(self):
        tc = tpc()
        self.X = [np.linspace(a, b, n)
            for a, b, n in zip(self.eq.a, self.eq.b, self.eq.n)]
        Y = self.eq.build_r0()
        self.Y0 = _renormalize_new(Y, self.eq.a, self.eq.b, self.eq.n)
        self.tc += tpc() - tc
