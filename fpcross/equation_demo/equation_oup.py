import numpy as np
from scipy.linalg import solve_lyapunov


from ..equation import Equation


class EquationOUP(Equation):
    """Ornstein-Uhlenbeck process, a special case of the FPE."""

    def __init__(self, d=3, e=1.E-4, is_full=False, name='OUP'):
        super().__init__(d, e, is_full, name)

        self.set_coef_rhs()
        self.set_grid(n=30, a=-5., b=+5.)
        self.set_grid_time(m=100, t=5.)

    def f(self, X, t):
        return -X @ self.coef_rhs.T

    def f1(self, X, t):
        shape = X.shape[:-1] + (1, )  # Tiling directions for a row.
        grads = np.tile(-np.diag(self.coef_rhs), shape)
        return grads

    def init(self):
        self.with_rs = True
        self.with_rt = True if self.d == 1 else False

        self.W = solve_lyapunov(self.coef_rhs, 2. * self.coef * np.eye(self.d))
        self.Wi = np.linalg.inv(self.W)
        self.Wd = np.linalg.det(self.W)

    def rs(self, X):
        r = np.exp(-0.5 * np.diag(X @ self.Wi @ X.T))
        r /= np.sqrt(2**self.d * np.pi**self.d * self.Wd)
        return r

    def rt(self, X, t):
        if self.d != 1:
            raise ValueError('It is valid only for 1D case')

        K = self.coef_rhs[0, 0]
        S = (1. - np.exp(-2. * K * t)) / 2. / K
        S += self.coef * np.exp(-2. * K * t)
        a = 2. * S
        r = np.exp(-X * X / a) / np.sqrt(np.pi * a)
        return r.reshape(-1)

    def set_coef_rhs(self, coef_rhs=None):
        """Set coefficient matrix for the rhs.

        Args:
            coef_rhs (np.ndarray): coefficient matrix for the rhs (rhs = -1 *
                coef_rhs * x). It is np.ndarray of the shape [d, d].

        """
        if coef_rhs is None:
            self.coef_rhs = np.eye(self.d)
        else:
            self.coef_rhs = coef_rhs



            
class EquationOUPUniv(EquationOUP):
    def __init__(self, *args, **kwargs):
        mu = kwargs.pop('mu', 0)
        self.set_coef_mu(mu)
        
        super().__init__(*args, **kwargs)


    def build(self, func, Y0=None, e=None):
        """Build the function on the Chebyshev grid related to equation.

        Args:
            func (function): function f(X) for interpolation, where X should be
                2D np.ndarray of the shape [samples, dimensions]. The function
                should return 1D np.ndarray of the length equals to "samples".
            Y0 (list): TT-tensor, which is the initial approximation for the
                TT-cross algorithm. If it is not set, then random TT-tensor of
                the TT-ranks equals "self.cross_r" will be used.
            e (float): convergence criterion for the TT-cross algorithm. If
                between iterations the relative rate of solution change is less
                than this value, then the operation of the algorithm will be
                interrupted. If is not set, then "self.e" option will be used.

        Returns:
            list: TT-Tensor with function values on the Chebyshev grid. If full
            format is used (see flag "is_full"), then the d-dimensional
            np.ndarray with function values on the Chebyshev grid will be
            returned.

        Note:
            You can use the "set_cross_opts" function to refine the parameters
            of the TT-cross method.

        """
        self.cross_info = {}
        self.cross_cache = {} if self.cross_with_cache else None

        if Y0 is None:
            Y0 = teneva.tensor_rand(self.n, self.cross_r)

        e = e or self.e
        a, b, n = teneva.grid_prep_opts(self.a, self.b, self.n)
        
        
        """
        Y = teneva.cheb_bld(func, self.a, self.b, self.n, eps=e, Y0=Y0,
            m=self.cross_m,
            e=e,
            nswp=self.cross_nswp,
            tau=self.cross_tau,
            dr_min=self.cross_dr_min,
            dr_max=self.cross_dr_max,
            tau0=self.cross_tau0,
            k0=self.cross_k0,
            info=self.cross_info,
            cache=self.cross_cache)
        """
            
        Y = teneva.cross(lambda I: func(teneva.ind_to_poi(I, a, b, n, 'uni')),
            Y0, self.cross_m, e, 
                         self.cross_nswp, self.cross_tau, 
                         self.cross_dr_min, self.cross_dr_max,
                         self.cross_tau0, self.cross_k0,
                         self.cross_info,
                         self.cross_cache)
        return teneva.truncate(Y, e)
        

        return Y
    

    def set_coef_mu(self, mu=0):
        self.coef_mu = mu
    
    
    
    def build_rs(self):
        """Build the stationary PDF r0(x) on the Chebyshev grid.

        Returns:
            list: TT-Tensor with function values on the Chebyshev grid. If full
            format is used (see flag "is_full"), then the d-dimensional
            np.ndarray with function values on the Chebyshev grid will be
            returned. If flag "with_rs" is not set, then it returns None.

        """
        if not self.with_rs:
            self.Ys = None
            return self.Ys

        func = self.rs
        self.cross_nswp = 100
        self.Ys = self.build(func, e=self.e/100)
        
        #self.Ys = self._renormalize(self.Ys)
        
        return self.Ys
    
    def _renormalize(self, Y):
        c = np.array([1.])
        for ci, a, b, n in zip(Y, self.a, self.b, self.n):
            c = c @ ci.sum(axis=1)
            c *= (b - a)/n
            
        self.s = c.item()
        return teneva.mul(1. / self.s, Y)
    
    
    def set_coef_mu(self, mu=0):
        self.coef_mu = mu
    
    def rs(self, X):
        self.log_M = -0.5 * np.diag( (X - self.coef_mu ) @ self.Wi @ (X.T - self.coef_mu ) )
        r = np.exp(self.log_M)
        Det = np.sqrt(2**self.d * np.pi**self.d * self.Wd)
        # self.Det = Det
        r /= Det
        return r


