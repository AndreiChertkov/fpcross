import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Equation(object):
    '''
    Class that represents stochastic differential equation.
    dx = f(t, x) dt + s(t, x) dW, fx = df / dx,
    where x is a d-dim vector, dW is a q-dim noise (Brownian motion),
    x0 is initial condition,
    r0 is initial distribution of r = r(t, x).
    '''

    def __init__(self, d=1, q=None, m=1):
        '''
        Constructor with parameters:
        d     - [int] dimension of coordinate
        q     - [int] dimension of noise (default =d)
        m     - [int] number of samples
        '''

        self.d = d or 1
        self.q = q or d
        self.m = m or 1

    def init_t_lim(self, t_min=0., t_max=1., t_poi=100):
        '''
        Set values for time limits:
        t_min - [float] start time
        t_max - [float] end time
        t_poi - [int] number of time steps
        '''

        self.t_min = t_min
        self.t_max = t_max
        self.t_poi = t_poi

        # Time step
        self.h = (self.t_max - self.t_min) / (self.t_poi - 1)

    def init_x_lim(self, x_min=0., x_max=1., x_poi=100):
        '''
        Set values for spatial variable limits:
        x_min - [float or list or d nd.array] minimum value
        x_max - [float or list or d nd.array] maximum value
        x_poi - [int or list or d nd.array] number of spatial points
        '''

        def to_arr(x):
            if isinstance(x, (int, float)): return list(np.ones(self.d) * x)
            if isinstance(x, (np.ndarray)): return list(x)
            return x

            if isinstance(x, (int, float)): return np.ones(self.d) * x
            if isinstance(x, (list)): return np.array(x)
            return x

        self.x_min = to_arr(x_min)
        self.x_max = to_arr(x_max)
        self.x_poi = [int(n) for n in to_arr(x_poi)]

        # X-grid (2d case!)
        self.Xg = np.linspace(x_min, x_max, x_poi)
        self.Xg1, self.Xg2 = np.meshgrid(self.Xg, self.Xg)
        self.Xg = np.array([self.Xg1, self.Xg2])
        self.Xg = self.Xg.reshape((2, -1))

    def init_funcs(self, s, f, fx, x0, r0, xr=None):
        '''
        Set equation functions:
        s(t, x)  - [d x q nd.array] noise coefficients
        f(t, x)  - [d or d x m nd.array] rhs function
        fx(t, x) - [d or d x m nd.array] rhs derivative
        x0(t, x) - [d nd.array] initial condition
        r0(t, x) - [float] initial distribution
        xr(t, w) - [d or d x m nd.array] (optional) real solution
        Function inputs:
            t - [float] current time
            x - [d x m nd.array] current coordinate
            w - [q x m nd.array] current Brownian motion value
        '''

        self._s = s
        self._f = f
        self._fx = fx
        self._xr = xr

        self.x0 = x0(self.t_min, None).reshape(self.d, 1)
        self.x0 = np.repeat(self.x0, self.m, axis=1)

        self.r0 = r0(self.t_min, self.Xg)

    def prep(self):
        '''
        Set all main variables to default state (t=t0, x=x0, w=0)
        (remove result of previous calculation).
        '''

        # Current values
        self.i = 0
        self.t = self.t_min
        self.x = self.x0.copy()
        self.r = self.r0
        self.w = 0.

        # Values for each time step
        self.T = np.linspace(self.t_min, self.t_max, self.t_poi)
        self.X = [self.x.copy()]
        self.R = [self.r]
        self.W = [self.w]
        self.Xr = [self.x0.copy()]

    def step(self, x, r):
        '''
        Add new time step.
        '''

        self.i+= 1
        self.t = self.T[self.i]
        self.x = x.copy().reshape(self.d, self.m)
        self.r = r
        self.w+= np.sqrt(self.h) * np.random.randn(self.d, self.m)

        if True:
            self.X.append(self.x.copy())
            self.R.append(self.r)
            self.W.append(self.w.copy())
            self.Xr.append(self.xr(self.t, self.w))

    def _func2samples(self, v):
        if len(v.shape) == 2: return v
        return np.repeat(v.reshape(self.d, 1), self.m, axis=1)

    def s(self, t, x):
        return self._func2samples(self._s(t, x))

    def f(self, t, x):
        return self._func2samples(self._f(t, x))

    def fx(self, t, x):
        return self._func2samples(self._fx(t, x))

    def xr(self, t, w):
        return self._func2samples(self._xr(t, w))

    def plot_x_vs_t(self, m=0):
        '''
        Plot calculation result x(t) for selected sample index m.
        '''

        if len(self.X) > 1:
            for i in range(self.d):
                plt.plot(self.T, [X[i, m] for X in self.X],
                    label='x_%d'%(i+1))

        if len(self.Xr) > 1:
            for i in range(self.d):
                plt.plot(self.T, [X[i, m] for X in self.Xr],
                    '--', label='x_%d (real)'%(i+1))

        plt.title('Solution')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend(loc='best')
        plt.show()

    def plot_x_err_vs_t(self, m=0):
        '''
        Plot calculation result x(t) error for selected sample index m.
        '''

        if len(self.X) > 1 and len(self.Xr) > 1:
            Xc = np.array([X[:, m] for X in self.X]).T
            Xr = np.array([X[:, m] for X in self.Xr]).T
            err = np.linalg.norm(Xc - Xr, axis=0)
            plt.plot(self.T, err)

        plt.semilogy()
        plt.title('Solution error')
        plt.xlabel('t')
        plt.ylabel('Error')
        plt.show()

    def plot_r(self):
        '''
        Plot distribution.
        '''

        x1 = self.Xg1
        x2 = self.Xg2
        r0 = self.r0.reshape((self.x_poi[0], self.x_poi[1]))
        rt = self.r.reshape((self.x_poi[0], self.x_poi[1]))

        print('min  : %-8.4f | %-8.4f'%(np.min(r0), np.min(rt)))
        print('max  : %-8.4f | %-8.4f'%(np.max(r0), np.max(rt)))
        print('mean : %-8.4f | %-8.4f'%(np.mean(r0), np.mean(rt)))

        fig = plt.figure(figsize=(10, 5))
        gs = mpl.gridspec.GridSpec(
            ncols=12, nrows=1, left=0.1, right=0.9, top=0.9, bottom=0.1)

        ax = fig.add_subplot(gs[0, :4])
        cs1 = ax.contourf(x1, x2, r0)
        ax.set_title('Initial spatial distribution (t=0)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        ax = fig.add_subplot(gs[0, 8:])
        cs2 = ax.contourf(x1, x2, rt)
        ax.set_title('Final spatial distribution (t)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        ax = fig.add_subplot(gs[0, 5:7])
        cb = plt.colorbar(cs1, cax=ax)
        # cb.add_lines(cs2)

        #fig.savefig('./figures/tmp.png', transparent=False, dpi=200, bbox_inches="tight", pad_inches=0.5)
        plt.show()
