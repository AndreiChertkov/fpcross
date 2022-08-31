import numpy as np
import teneva


class Equation:
    def __init__(self, d, e, is_full=False, name='Equation'):
        """Class that represents the Fokker-Planck equation.

        Args:
            d (int): number of dimensions.
            e (float): desired accuracy of approximation. It will be used for
                all truncations of the TT-tensors. Note that this parameter is
                not used if "is_full" flag is set.
            is_full (bool): if flag is set, then full (numpy) format will be
                used instead of the TT-format. Note that the full format may be
                used only for small dimension numbers ("d").
            name (str): optional display name for the equation.

        """
        self.d = int(d)
        self.e = float(e)
        self.is_full = bool(is_full)
        self.name = str(name)

        self.set_coef()
        self.set_coef_pdf()
        self.set_cross_opts()
        self.set_grid()
        self.set_grid_time()

        self.Ys = None
        self.Yt = None

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

        """
        self.cross_info = {}
        self.cross_cache = {} if self.cross_with_cache else None

        if Y0 is None:
            Y0 = teneva.tensor_rand(self.n, self.cross_r)

        if self.is_full:
            Y = teneva.cheb_bld_full(func, self.a, self.b, self.n)
        else:
            Y = teneva.cheb_bld(func, self.a, self.b, self.n, eps=self.e, Y0=Y0,
                m=self.cross_m,
                e=e or self.e,
                nswp=self.cross_nswp,
                tau=self.cross_tau,
                dr_min=self.cross_dr_min,
                dr_max=self.cross_dr_max,
                tau0=self.cross_tau0,
                k0=self.cross_k0,
                info=self.cross_info,
                cache=self.cross_cache)

        return Y

    def build_r0(self):
        """Build the initial PDF r(x, 0) on the Chebyshev grid.

        Returns:
            list: TT-Tensor with function values on the Chebyshev grid. If full
                format is used (see flag "is_full"), then the d-dimensional
                np.ndarray with function values on the Chebyshev grid will be
                returned.

        Note:
            The function specified in the "r0" method is used.

        """
        func = self.r0
        self.Y0 = self.build(func)
        return self.Y0

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
        self.Ys = self.build(func, e=self.e/100)
        return self.Ys

    def build_rt(self, t):
        """Build the time-dependent PDF r(x, t) on the Chebyshev grid.

        Args:
            t (float): the current value of time.

        Returns:
            list: TT-Tensor with function values on the Chebyshev grid. If full
                format is used (see flag "is_full"), then the d-dimensional
                np.ndarray with function values on the Chebyshev grid will be
                returned. If flag "with_rt" is not set, then it returns None.

        """
        if not self.with_rt:
            self.Yt = None
            return self.Yt

        func = lambda X: self.rt(X, t)
        self.Yt = self.build(func, e=self.e/100)
        return self.Yt

    def callback(self, solver):
        """Callback function, which is called by solver after each time step.

        Args:
            solver (FPCross): Fokker-Planck solver class instance.

        """
        return

    def f(self, X, t):
        """The function computes the rhs of equation.

        Args:
            X (np.ndarray): the set of spatial points. It is np.ndarray of the
                shape [samples, dimensions].
            t (float): the current value of time.

        Returns:
            np.ndarray: the values of the vector-values rhs for all samples (it
            is 2D array of the shape [samples, dimensions]).

        """
        return np.zeros(X.shape)

    def f1(self, X, t):
        """The function computes the derivatives for rhs of equation.

        Args:
            X (np.ndarray): the set of spatial points. It is np.ndarray of the
                shape [samples, dimensions].
            t (float): the current value of time.

        Returns:
            np.ndarray: the values of the "diagonal" partial derivatives for
            the rhs ( d f_1 / d x_1, d f_2 / d x_2, ..., d f_d / d x_d) for all
            samples (it is 2D array of the shape [samples, dimensions]).

        """
        return np.zeros(X.shape)

    def init(self):
        """The function is used to initialize the equation parameters.

        Run it before the solution by FPCross solver but after the setting all
        equation parameters ("set_coef", "set_grid", etc.).

        """
        self.with_rs = False
        self.with_rt = False

    def r0(self, X):
        """The function computes initial PDF r(x, 0).

        Args:
            X (np.ndarray): the set of spatial points. It is np.ndarray of the
                shape [samples, dimensions].

        Returns:
            np.ndarray: the values of the initial PDF r(x, 0) for all samples
            (it is 1D array of the shape [samples]).

        Note:
            This function can be changed in a child class. Here we use the
            Gaussian distribution with parameter "coef_pdf", which is defined
            in "set_coef_pdf" function.

        """
        a = 2. * self.coef_pdf
        r = np.exp(-np.sum(X*X, axis=1) / a)
        r /= (np.pi * a)**(self.d/2)
        return r

    def rs(self, X):
        """The function computes stationary PDF r0(x).

        Args:
            X (np.ndarray): the set of spatial points. It is np.ndarray of the
                shape [samples, d].

        Returns:
            np.ndarray: the values of the stationary PDF r0(x) for all samples
            (it is 1D array of the shape [samples]).

        Note:
            This function can be defined in a child class. In this case, the
            flag "self.with_rs" must also be set manually to True value (e.g.,
            inside the "init" function). In this case, the Fokker-Planck solver
            will automatically check the accuracy of the received solution.

        """
        return

    def rt(self, X, t):
        """The function computes time-dependent PDF r(x, t).

        Args:
            X (np.ndarray): the set of spatial points. It is np.ndarray of the
                shape [samples, dimensions].
            t (float): the current value of time.

        Returns:
            np.ndarray: the values of the time-dependent PDF r(x, t) for all
            samples (it is 1D array of the shape [samples]).

        Note:
            This function can be defined in a child class. In this case, the
            flag "self.with_rt" must also be set manually to True value (e.g.,
            inside the "init" function). In this case, the Fokker-Planck solver
            will automatically check the accuracy of the received solution.

        """
        return

    def set_coef(self, coef=0.5):
        """Set diffusion coefficient.

        Args:
            coef (float): scalar diffusion coefficient.

        """
        self.coef = coef

    def set_coef_pdf(self, coef_pdf=1.):
        """Set coefficient for initial PDF Gaussian distribution.

        Args:
            coef_pdf (float): coefficient for initial PDF.

        """
        self.coef_pdf = coef_pdf

    def set_cross_opts(self, m=None, e=None, nswp=10, tau=1.1, dr_min=1, dr_max=2, tau0=1.05, k0=100, with_cache=True, r=1):
        """Set parameters for the TT-CROSS method.

        See "https://teneva.readthedocs.io/code/core/cross.html" with a
        detailed description of the parameters. In addition to the parameters
        described on this page, this function also accepts "with_cache" (if
        flag is True, then cache for TT-CROSS will be used), "r" (TT-rank for
        initial approximation).

        """
        self.cross_m = m
        self.cross_e = e
        self.cross_nswp = nswp
        self.cross_tau = tau
        self.cross_dr_min = dr_min
        self.cross_dr_max = dr_max
        self.cross_tau0 = tau0
        self.cross_k0 = k0
        self.cross_with_cache = with_cache
        self.cross_r = r

    def set_grid(self, n=10, a=-1., b=+1.):
        """Set spatial grid parameters for the discretization.

        Args:
            n (int, float, list, np.ndarray): grid size for each dimension
                (list or np.ndarray of length "d"). It may be also int/float,
                then the size for each dimension will be the same.
            a (float, list, np.ndarray): lower bounds for each dimension (list
                or np.ndarray of length "d"). It may be also float, then the
                lower bounds for each dimension will be the same.
            b (float, list, np.ndarray): upper bounds for each dimension (list
                or np.ndarray of length "d"). It may be also float, then the
                upper bounds for each dimension will be the same.

        Note:
            Only "unfirom" grids are supported now (the size for each dimension
            should be the same), i.e., n / a / b should be int/float.

        """
        if isinstance(n, (int, float)):
            n = np.ones(self.d, dtype=int) * int(n)
        else:
            raise ValueError('Only "unfirom" grids are supported now')
        self.n = np.asanyarray(n, dtype=int)
        if len(self.n) != self.d:
            raise ValueError('Invalid length of n')

        if isinstance(a, (int, float)):
            a = [a] * self.d
        else:
            raise ValueError('Only "unfirom" grids are supported now')
        self.a = np.asanyarray(a, dtype=float)
        if len(self.a) != self.d:
            raise ValueError('Invalid length of a')

        if isinstance(b, (int, float)):
            b = [b] * self.d
        else:
            raise ValueError('Only "unfirom" grids are supported now')
        self.b = np.asanyarray(b, dtype=float)
        if len(self.b) != self.d:
            raise ValueError('Invalid length of b')

    def set_grid_time(self, m=100, t=1.):
        """Set time grid parameters for the discretization.

        Args:
            m (int, float): grid size (i.e., the number of time steps).
            t (float): upper bound for time grid.

        """
        self.m = int(m)
        self.t = t
        self.h = self.t / self.m
