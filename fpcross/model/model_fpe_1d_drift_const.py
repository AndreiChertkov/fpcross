import numpy as np

from .model import Model as ModelBase

class Model(ModelBase):

    def name(self):
        return 'fpe_1d_drift_const'

    def repr(self):
        return r'd r(x,t) / d t = D d^2 r(x,t) / d x^2 - v d r(x,t) / d x'

    def desc(self):
        return 'One-dimensional Focker Planck equation with the constant drift'

    def tags(self):
        return ['FPE', '1D', 'analytic']

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
            'v': {
                'name': 'Drift',
                'desc': 'Constant drift value',
                'dflt': 0.02,
                'type': 'float',
                'frmt': '%8.4f',
            },
        }

    def coms(self):
        return [
            r'The final solution is not vanish on the boundary, hence we have significant integral error on the grid. At the same time, on the inner grid points solution is very accurate.',
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
                f(x, t) \equiv v,
                \quad
                \rho_0(x) = \frac{1}{\sqrt{2 \pi s}}\exp{\left[-\frac{x^2}{2s}\right]}.
            $$

            This equation has exact solution ([see this paper](http://www.icmp.lviv.ua/journal/zbirnyk.73/13002/art13002.pdf); note that there is a typo in the paper for this formula: $\pi$ is missed!)
            $$
                \rho(x, t, x_0) =
                    \frac{1}{\sqrt{4 \pi D t}}
                    \exp{\left[
                        - \frac
                            {
                                \left( x - x_0 - v t \right)^2
                            }
                            {
                                4 D t
                            }
                    \right]},
            $$
            where $x_0$ is an initial condition.

            We can rewrite the solution $\rho(x, t, x_0)$ in terms of the initial PDF $\rho_0(x)$ as
            $$
                \rho(x, t) = \int_{-\infty}^{\infty}
                    \rho(x, t, x_0) \rho_0(x_0) \, d x_0,
            $$
            which after accurate computations leads to the following analytic solution
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
                                (x - vt)^2
                            }
                            {
                                2  s + 4 D t
                            }
                    \right] },
            $$
            and the stationary solution ($t \rightarrow \infty$) is
            $$
                \rho_{stat}(x) = 0.
            $$
        '''

    def d(self):
        return 1

    def D(self):
        D = self._D

        return D

    def f0(self, X, t):
        v = self._v

        return v * np.ones(X.shape)

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
        v = self._v

        a = 2. * s + 4. * D * t

        r = np.exp(-(X - v * t)**2 / a) / np.sqrt(np.pi * a)

        return r.reshape(-1)

    def with_rt(self):
        return True