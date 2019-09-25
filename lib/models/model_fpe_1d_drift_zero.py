import numpy as np

from model_ import Model as ModelBase

name = 'fpe_1d_drift_zero'
desc = 'One-dimensional Focker Planck equation with the zero drift'
tags = ['FPE', '1D', 'analyt', 'time-diffusion']
info = {
    'markdown': r'''

<div class="head0">
    <div class="head0__name">
        Model problem
    </div>
    <div class="head0__note">
        One-dimensional Focker Planck equation with the zero drift
    </div>
</div>

<div class="head2">
    <div class="head2__name">
        Parameters
    </div>
    <div class="head2__note">
        <ul>
            <li>$s$ - variance of the initial condition (float, default $= 0.1$)</li>
            <li>$D_c$ - diffusion coefficient (float, default $= 0.02$)</li>
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
        \right] },
$$
and the stationary solution ($t \rightarrow \infty$) is
$$
    \rho_{stat}(x) = 0.
$$

<div class="note">
    Since interpolation is not required for the case of the zero drift ($f \equiv 0$), but our solver calculates it by design, then it is expected to operate much slower than another simple solvers.
</div>

<div class="end"></div>
    '''
}

class Model(ModelBase):

    def __init__(self):
        super().__init__(name, desc, tags, info)

    def init(self, s=None, D=None):
        self.d = 1
        self._set('s', s, 0.1)
        self._set('D', D, 0.02)

    def _d0(self):
        return self.D

    def _f0(self, x, t):
        return np.zeros(x.shape)

    def _f1(self, x, t):
        return np.zeros(x.shape)

    def _r0(self, x):
        a = 2. * self.s
        r = np.exp(-x * x / a) / np.sqrt(np.pi * a)
        return r.reshape(-1)

    def _rt(self, x, t):
        a = 2. * self.s + 4. * self.D * t
        r = np.exp(-x * x / a) / np.sqrt(np.pi * a)
        return r.reshape(-1)

    def _rs(self, x):
        r = np.zeros(x.shape)
        return r.reshape(-1)

    def _with_rt(self):
        return True

    def _with_rs(self):
        return False
