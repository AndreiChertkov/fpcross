import numpy as np
from scipy.linalg import solve_lyapunov

from .model import Model as ModelBase

class Model(ModelBase):

    def name(self):
        return 'fpe-oup'

    def repr(self):
        return r'd r(x,t) / d t = D \Delta r(x,t) + A div[ x r(x,t) ]'

    def desc(self):
        return 'Multidimensional Focker Planck equation (Ornstein–Uhlenbeck process)'

    def tags(self):
        return ['FPE', 'ND', 'analytic', 'analyt-stationary', 'OUP']

    def pars(self):
        return {
            'd': {
                'name': 'Dimension',
                'desc': 'Spatial dimension',
                'dflt': 1,
                'type': 'int',
                'frmt': '%3d',
            },
            's': {
                'name': 'Initial variance',
                'desc': 'Variance of the initial condition',
                'dflt': 1.,
                'type': 'float',
                'frmt': '%8.4f',
            },
            'D': {
                'name': 'Diffusion coefficient',
                'desc': 'Scalar diffusion coefficient',
                'dflt': 0.5,
                'type': 'float',
                'frmt': '%8.4f',
            },
            'A': {
                'name': 'Drift',
                'desc': 'Constant drift coefficients ([d x d] matrix)',
                'dflt': lambda vals: np.eye(vals.get('d', 1)),
                'type': 'ndarray',
                'frmt': '%s',
            },
        }

    def coms(self):
        return [
            r'The multivariate Ornstein–Uhlenbeck process is mean-reverting (the solution tends to its long-term mean $\mu$ as time $t$ tends to infinity) if if all eigenvalues of $A$ are positive and this process at any time is a multivariate normal random variable.',

            r'We do not construct analytic solution for this multidimensional case, but use comparison with known stationary solution. The corresponding error will depend on the maximum value for the used time grid.',
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
                D(x, t) \equiv D I,
            $$
            and
            $$
                \quad
                x \in \Omega,
                \quad
                \rho(x, t) |_{\partial \Omega} \approx 0,
                \quad
                f(x, t) = A (\mu - x),
                \quad
                \mu \equiv 0,
                \quad
                \rho_0(x) =
                    \frac{1}{\left(2 \pi s \right)^{\frac{d}{2}}}
                    \exp{\left[-\frac{|x|^2}{2s}\right]}.
            $$

            This equation has stationary solution ($t \rightarrow \infty$)
            $$
                \rho_{stat}(x) =
                    \frac
                    {
                        exp \left[ -\frac{1}{2} x^{\top} W^{-1} x \right]
                    }
                    {
                        \sqrt{(2 \pi)^d det(W)}
                    },
            $$
            where matrix $W$ is solution of the matrix equation
            $$
                A W + W A^{\top} = 2 D.
            $$
        '''

    def prep(self):
        d = self.d()
        D = self.D()
        A = self._A

        self._W = solve_lyapunov(A, 2. * D * np.eye(d))
        self._Wi = np.linalg.inv(self._W)
        self._Wd = np.linalg.det(self._W)

    def d(self):
        return self._d

    def D(self):
        D = self._D

        return D

    def f0(self, X, t):
        A = self._A

        return -A @ X

    def f1(self, X, t):
        A = self._A

        return -A @ np.ones(X.shape)

    def r0(self, X):
        d = self.d()
        s = self._s

        a = 2. * s

        r = np.exp(-np.sum(X*X, axis=0) / a) / (np.pi * a)**(d/2)

        return r.reshape(-1)

    def rs(self, X):
        d = self.d()
        Wi = self._Wi
        Wd = self._Wd

        r = np.exp(-0.5 * np.diag(X.T @ Wi @ X))
        r/= np.sqrt(2**d * np.pi**d * Wd)

        return r

    def with_rs(self):
        return True
