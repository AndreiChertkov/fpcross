import numpy as np

from model_ import Model as ModelBase

name = 'fpe-1d-oup'
desc = 'One-dimensional Focker Planck equation (Ornstein–Uhlenbeck process)'
tags = ['FPE', '1D', 'OUP']
info = {
    'latex': r'''

<div class="head0">
    <div class="head0__name">
        Model problem
    </div>
    <div class="head0__note">
        One-dimensional Focker Planck equation with linear drift (Ornstein–Uhlenbeck process)
    </div>
</div>

<div class="head2">
    <div class="head2__name">
        Parameters
    </div>
    <div class="head2__note">
        <ul>
            <li>$s$ - variance of the initial condition (float, default $= 1$)</li>
            <li>$D_c$ - diffusion coefficient (float, default $= 0.5$)</li>
            <li>$A$ - constant drift coefficient (float, default $= 1$)</li>
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
    d = 1,
    \quad
    x \in \Omega,
    \quad
    \rho(x, t) |_{\partial \Omega} \approx 0,
    \quad
    f(x, t) = A (\mu - x),
    \quad
    \mu \equiv 0,
    \quad
    \rho_0(x) = \frac{1}{\sqrt{2 \pi s}}\exp{\left[-\frac{x^2}{2s}\right]}.
$$

This equation has exact solution for the $d$-dimensional case ([see this paper](https://advancesindifferenceequations.springeropen.com/articles/10.1186/s13662-019-2214-1))
$$
    \rho(x, t, x_0) =
        \frac{1}{\sqrt{ | 2 \pi \Sigma(t) | }}
        exp \left[
            -\frac{1}{2} (x - M(t, x_0))^T \Sigma^{-1}(t) (x - M(t, x_0))
        \right],
$$
where $x_0$ is an initial condition and
$$
    M(t, x_0) = e^{-A t} x_0 + \left( I - e^{-A t} \right) \mu,
$$
$$
    \Sigma(t) = \int_0^t e^{A (s-t)} S S^T e^{A^T (s-t)} \, d  s,
$$
or in a more simple form for the case $d=1$
$$
    \rho(x, t, x_0) =
        \frac{1}{\sqrt{ | 2 \pi \Sigma(t) | }}
        exp \left[
            -\frac{(x - M(t, x_0))^2}{2 \Sigma(t)}
        \right],
    \quad
    M(t, x_0) = e^{-A t} x_0,
    \quad
    \Sigma(t) = \frac{1 - e^{-2 A t}}{2 A}.
$$

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
                \sqrt{2 \pi \left( \Sigma(t) + s e^{-2 A t} \right)}
            }
        \exp{\left[
            - \frac
                {
                    x^2
                }
                {
                    2 \left( \Sigma(t) + s e^{-2 A t} \right)
                }
        \right]},
$$
and the stationary solution ($t \rightarrow \infty$) is
$$
    \rho_{stat}(x) =
        \sqrt{
            \frac{A}{\pi}
        }
        e^{-A x^2}.
$$

<div class="note">
    The Ornstein–Uhlenbeck process is mean-reverting (the solution tends to its long-term mean $\mu$ as time $t$ tends to infinity) if $A > 0$ and this process at any time is a normal random variable.
</div>

<div class="end"></div>
    '''
}

class Model(ModelBase):

    def __init__(self):
        super().__init__(name, desc, tags, info)

    def init(self, s=None, A=None, D=None):
        self.d = 1
        self._set('s', s, 1.)
        self._set('D', D, 0.5)
        self._set('A', A, 1.)
        
    def _d0(self):
        return self.D

    def _f0(self, x, t):
        return -self.A * x

    def _f1(self, x, t):
        return -self.A * np.ones(x.shape)

    def _r0(self, x):
        a = 2. * self.s
        r = np.exp(-x * x / a) / np.sqrt(np.pi * a)
        return r.reshape(-1)

    def _rt(self, x, t):
        def _xc(t):
            return (1. - np.exp(-2. * self.A * t)) / 2. / self.A

        S = _xc(t) + self.s * np.exp(-2. * self.A * t)
        a = 2. * S
        r = np.exp(-x * x / a) / np.sqrt(np.pi * a)
        return r.reshape(-1)

    def _rs(self, x):
        a = 1. / self.A
        r = np.exp(-x * x / a) / np.sqrt(np.pi * a)
        return r.reshape(-1)

    def _with_rt(self):
        return True

    def _with_rs(self):
        return True
