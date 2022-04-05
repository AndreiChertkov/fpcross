import numpy as np


from ..equation import Equation


class EquationDif(Equation):
    """Simple diffusion problem, a special case of the FPE."""

    def __init__(self, d=3, e=1.E-6, is_full=False, name='Dif'):
        super().__init__(d, e, is_full, name)

        self.set_grid(n=20, a=-10., b=+10.)
        self.set_grid_time(m=100, t=2.5)

    def f(self, X, t):
        return np.zeros(X.shape)

    def f1(self, X, t):
        return np.zeros(X.shape)

    def init(self):
        self.with_rs = False
        self.with_rt = True

    def rt(self, X, t):
        a = 2. * self.coef_pdf + 4. * self.coef * t
        r = np.exp(-np.sum(X*X, axis=1) / a)
        r /= (np.pi * a)**(self.d/2)
        return r
