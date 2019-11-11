import numpy as np

from .model import Model as ModelBase

class Model(ModelBase):

    def name(self):
        return 'fpe_1d_drift_zero'

    def repr(self):
        return r'd r(x,t) / d t = D d^2 r(x,t) / d x^2'

    def desc(self):
        return 'One-dimensional Focker-Planck equation with the zero drift'

    def tags(self):
        return ['FPE', '1D', 'analytic', 'time-diffusion']

    def pars(self):
        return {
            's': {
                'name': 'Initial variance',
                'desc': 'Variance of the initial condition',
                'dflt': 0.1,
                'type': 'float',
                'frmt': '%8.4f',
            },
            'D': {
                'name': 'Diffusion coefficient',
                'desc': 'Scalar diffusion coefficient',
                'dflt': 0.02,
                'type': 'float',
                'frmt': '%8.4f',
            },
        }

    def coms(self):
        return [
            r'Since interpolation is not required for the case of the zero drift ($f \equiv 0$), but our solver calculates it by design, then it is expected to operate much slower than another simple solvers.',
        ]

    def text(self):
        return r'''
            Consider
            $$
                d x = f(x, t) \, dt + S(x, t) \, d \beta,
                \quad
                d \beta \, d \beta^{\top} = Q(t) dt,
                \quad
                x(0) = x_0 \sim \rho(x, 0) = \rho_0 (x),
            $$
            $$
                \frac{\partial \rho(x, t)}{\partial t} =
                    \sum_{i=1}^d \sum_{j=1}^d
                        \frac{\partial^2}{\partial x_i \partial x_j}
                        \left[ D_{ij}(x, t) \rho(x, t) \right]
                    - \sum_{i=1}^d
                        \frac{\partial}{\partial x_i}
                        \left[ f_i(x, t) \rho(x, t) \right],
                \quad
                 D(x, t) = \frac{1}{2} S(x, t) Q(t) S(x, t)^{\top},
            $$
            where spatial $d$-dimensional ($d \ge 1$) variable $x \in R^d$ has probability density function (PDF) $\rho(x, t)$, $\beta$ is Brownian motion of dimension $q$ ($q \ge 1$, and we assume below that $q = d$), $f(x, t) \in R^d$ is a vector-function, $S(x, t) \in R^{d \times q}$ and $Q(t) \in R^{q \times q}$ are matrix-functions and $D(x, t) \in R^{d \times d}$ is a diffusion tensor.

            Let
            $$
                Q(t) \equiv I,
                \,
                S(x, t) \equiv \sqrt{2 D_c} I
                \implies
                D(x, t) \equiv D_c I,
            $$
            and
            $$
                d = 1,
                \quad
                x \in \Omega,
                \quad
                \rho(x, t) |_{\partial \Omega} \approx 0,
                \quad
                f(x, t) \equiv 0,
                \quad
                \rho_0(x) = \frac{1}{\sqrt{2 \pi s}}\exp{\left[-\frac{x^2}{2s}\right]}.
            $$

            It can be shown that the analytic solution is
            $$
                \rho(x, t) =
                    \frac
                        {
                            1
                        }
                        {
                            \sqrt{2 \pi s + 4 \pi D t}
                        }
                    \exp{ \left[
                        - \frac
                            {
                                x^2
                            }
                            {
                                2  s + 4 D t
                            }
                    \right] }.
            $$
        '''

    def d(self):
        return 1

    def D(self):
        D = self._D

        return D

    def f0(self, X, t):
        return np.zeros(X.shape)

    def f1(self, X, t):
        return np.zeros(X.shape)

    def r0(self, X):
        s = self._s

        a = 2. * s

        r = np.exp(-X * X / a) / np.sqrt(np.pi * a)

        return r.reshape(-1)

    def rt(self, X, t):
        D = self.D()
        s = self._s

        a = 2. * s + 4. * D * t

        r = np.exp(-X * X / a) / np.sqrt(np.pi * a)

        return r.reshape(-1)

    def with_rt(self):
        return True
