from os import path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc

class Equation(object):
    '''
    Class that represents stochastic differential equation.
    dx = f(t, x) dt + s(t, x) dW, fx = df / dx,
    where x is a d-dim vector, dW is a q-dim noise (Brownian motion),
    x0 is initial condition, r0 is initial distribution of r = r(t, x).
    '''

    def __init__(self, d=2, q=None, m=1):
        '''
        Constructor with parameters:
        d     - [int] dimension of coordinate (default: 2)
        q     - [int] dimension of noise (default: d)
        m     - [int] number of samples (default: 1)
        '''

        self.d = d or 1
        self.q = q or d
        self.m = m or 1

        if self.d != 2:
            raise NotImplementedError('Spatial dimension should be 2')

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
        x_min - [float or list or d nd.array] minimum value for each dim.
        x_max - [float or list or d nd.array] maximum value for each dim.
        x_poi - [int or list or d nd.array] number of points for each dim.
        '''

        def to_list(x):
            if isinstance(x, (int, float)): return list(np.ones(self.d) * x)
            if isinstance(x, (np.ndarray)): return list(x)
            return x

        self.x_min = to_list(x_min)
        self.x_max = to_list(x_max)
        self.x_poi = [int(n) for n in to_list(x_poi)]

        # X-grid (2d case!)
        self.Xg = np.linspace(x_min, x_max, x_poi)
        self.Xg1, self.Xg2 = np.meshgrid(self.Xg, self.Xg)
        self.Xg = np.array([self.Xg1, self.Xg2])
        self.Xg = self.Xg.reshape((2, -1))

        # self.m =

    def init_funcs(self, s, f, fx, x0, r0, xr=None):
        '''
        Set equation functions:
        s(t, x)  - [d x q nd.array] noise coefficients matrix
                   t - [float] current time
                   x - [d x m nd.array] current coordinate
        f(t, x)  - [d or d x 1 nd.array] rhs function
        fx(t, x) - [d x d nd.array] rhs derivative
        x0(t, x) - [d or d x 1 nd.array] initial condition
        r0(t, x) - [x.shape nd.array] initial distribution
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
            if self._xr:
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

    def pres(self):
        '''
        Present general info.
        '''

        r0 = self.r0
        rt = self.r

        print('r min  : t=min %-8.4f | t=max %-8.4f'%(np.min(r0), np.min(rt)))
        print('r max  : t=min %-8.4f | t=max %-8.4f'%(np.max(r0), np.max(rt)))
        print('r mean : t=min %-8.4f | t=max %-8.4f'%(np.mean(r0), np.mean(rt)))

    def plot_x(self, m=0):
        '''
        Plot calculation result x(t) for selected sample index m.
        '''

        if len(self.X) <= 1 and len(self.Xr) <= 1: return
        if len(self.Xr) <= 1:
            return self.plot_x_vs_t(None, m)

        fig = plt.figure(figsize=(10, 5))
        gs = mpl.gridspec.GridSpec(
            ncols=12, nrows=1, left=0.1, right=0.9, top=0.9, bottom=0.1)

        self.plot_x_vs_t(fig.add_subplot(gs[0, :5]), m)
        self.plot_x_err_vs_t(fig.add_subplot(gs[0, 7:]), m)

    def plot_x_vs_t(self, ax=None, m=0):
        '''
        Plot calculation result x(t) for selected sample index m.
        '''

        if not ax: ax = plt.figure(figsize=(5, 5)).add_subplot(111)
        if len(self.X) <= 1 and len(self.Xr) <= 1: return

        if len(self.X) > 1:
            for i in range(self.d):
                ax.plot(self.T, [X[i, m] for X in self.X],
                    label='x_%d'%(i+1))

        if len(self.Xr) > 1:
            for i in range(self.d):
                ax.plot(self.T, [X[i, m] for X in self.Xr],
                    '--', label='x_%d (real)'%(i+1))

        ax.set_title('Solution')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.legend(loc='best')

    def plot_x_err_vs_t(self, ax=None, m=0):
        '''
        Plot calculation result x(t) error for selected sample index m.
        '''

        if len(self.X) <= 1 or len(self.Xr) <= 1: return
        if not ax: ax = plt.figure(figsize=(5, 5)).add_subplot(111)

        Xc = np.array([X[:, m] for X in self.X]).T
        Xr = np.array([X[:, m] for X in self.Xr]).T
        err = np.linalg.norm(Xc - Xr, axis=0)
        ax.plot(self.T, err)
        ax.semilogy()
        ax.set_title('Solution error')
        ax.set_xlabel('t')
        ax.set_ylabel('Error')

    def plot_r(self):
        '''
        Plot distribution r at initial and final time moments.
        '''

        x1 = self.Xg1
        x2 = self.Xg2
        r0 = self.r0.reshape((self.x_poi[0], self.x_poi[1]))
        rt = self.r.reshape((self.x_poi[0], self.x_poi[1]))

        fig = plt.figure(figsize=(10, 6))
        gs = mpl.gridspec.GridSpec(
            ncols=4, nrows=2, left=0.01, right=0.99, top=0.99, bottom=0.01,
            wspace=0.4, hspace=0.3,
            width_ratios=[1, 1, 1, 1], height_ratios=[6, 1])

        ax = fig.add_subplot(gs[0, :2])
        ct1 = ax.contourf(x1, x2, r0)
        ax.set_title('Initial spatial distribution (t=0)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        ax = fig.add_subplot(gs[0, 2:])
        ct2 = ax.contourf(x1, x2, rt)
        ax.set_title('Final spatial distribution (t)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        ax = fig.add_subplot(gs[1, 1:3])
        cb = plt.colorbar(ct1, cax=ax, orientation='horizontal')
        # cb.add_lines(ct2)

        #fig.savefig('./figures/tmp.png', transparent=False, dpi=200, bbox_inches="tight", pad_inches=0.5)
        plt.show()

    def anim_r(self, fffolder=None, delt=200):
        '''
        Build animation for calculated distribution r as function of time t.
        '''

        if fffolder:
            ffpath = path.join(fffolder, 'ffmpeg')
            plt.rcParams['animation.ffmpeg_path'] = ffpath

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        def run(i):
            t = self.T[i]
            r = self.R[i].reshape((self.x_poi[0], self.x_poi[1]))
            x1 = self.Xg1
            x2 = self.Xg2

            ax.clear()
            ax.set_title('Spatial distribution (t=%f)'%t)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ct = ax.contourf(x1, x2, r)
            return (ct,)

        anim = animation.FuncAnimation(
            fig, run, frames=len(self.T), interval=delt, blit=False)
        return anim.to_html5_video()
