import numpy as np
from scipy.linalg import solve_lyapunov

from model_ import Model as ModelBase

name = 'fpe-oup'
desc = 'Multidimensional Focker Planck equation (Ornstein–Uhlenbeck process)'
tags = ['FPE', 'OUP', 'analyt-stat']
info = {
    'markdown': r'''

<div class="head0">
    <div class="head0__name">
        Model problem
    </div>
    <div class="head0__note">
        Multidimensional Focker Planck equation with linear drift (Ornstein–Uhlenbeck process)
    </div>
</div>

<div class="head2">
    <div class="head2__name">
        Parameters
    </div>
    <div class="head2__note">
        <ul>
            <li>$d$ - spatial dimension (int, default $= 1$)</li>
            <li>$s$ - variance of the initial condition (float, default $= 1$)</li>
            <li>$D_c$ - diffusion coefficient (float, default $= 0.5$)</li>
            <li>$A$ - constant drift matrix (array $d \times d$ of float, default $= I_d$)</li>
        </ul>
    </div>
</div>

<div class="head1">
    <div class="head1__name">
        Description
    </div>
</div>

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

<div class="note">
    The multivariate Ornstein–Uhlenbeck process is mean-reverting (the solution tends to its long-term mean $\mu$ as time $t$ tends to infinity) if if all eigenvalues of $A$ are positive and this process at any time is a multivariate normal random variable.
</div>

<div class="note">
    We do not construct analytic solution for this multidimensional case, but use comparison with known stationary solution. The corresponding error will depend on the maximum value for the used time grid.
</div>

<div class="end"></div>
    '''
}

class Model(ModelBase):

    def __init__(self):
        super().__init__(name, desc, tags, info)

    def init(self, d=None, s=None, D=None, A=None):
        self._set('d', d, 1)
        self._set('s', s, 1.)
        self._set('D', D, 0.5)
        self._set('A', A, np.eye(self.d))

        self.W = solve_lyapunov(self.A, 2. * self.D * np.eye(self.d))
        self.Wi = np.linalg.inv(self.W)
        self.Wd = np.linalg.det(self.W)

    def _d0(self):
        return self.D

    def _f0(self, x, t):
        return -self.A  @ x

    def _f1(self, x, t):
        return -self.A @ np.ones(x.shape)

    def _r0(self, x):
        a = 2. * self.s
        r = np.exp(-np.sum(x*x, axis=0) / a) / (np.pi * a)**(self.d/2)
        return r.reshape(-1)

    def _rs(self, x):
        r = np.exp(-0.5 * np.diag(x.T @ self.Wi @ x))
        r/= np.sqrt(2**self.d * np.pi**self.d * self.Wd)
        return r

    def _with_rt(self):
        return False

    def _with_rs(self):
        return True
