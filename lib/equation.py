import numpy as np
import matplotlib.pyplot as plt

class Equation(object):
    '''
    Class that represents stochastic differential equation.
    dx = f(x, t) dt + s(x, t) dW, fx = df / dx,
    where x is a d-dim vector, dW is a q-dim noise (Brownian motion),
    x0 is initial condition.
    '''

    def __init__(self, x0, d=1, q=None, m=1, t_min=0., t_max=1., t_poi=100):
        '''
        Constructor with parameters:
        x0    - [d or d x 1 nd.array] initial condition
        d     - [int] dimension of coordinate
        q     - [int] dimension of noise
        m     - [int] number of samples
        t_min - [float] start time
        t_max - [float] end time
        t_poi - [int] number of time steps
        '''

        self.d = d or 1
        self.q = q or d
        self.m = m or 1

        self.x0 = np.repeat(x0.reshape(self.d, 1), self.m, axis=1)

        self.t_min = t_min
        self.t_max = t_max
        self.t_poi = t_poi

        self.h = (t_max - t_min) / (t_poi - 1)

        self.init()

    def prep(self, s, f, fx, xr=None):
        '''
        Set equation functions:
        s(t, x)  - noise coefficients
        f(t, x)  - rhs function
        fx(t, x) - rhs derivative
        xr(t, w) - (optional) real solution
        Function inputs:
            t - [float] current time
            x - [d x m nd.array] current coordinate
            w - [q x m nd.array] current Brownian motion value
        '''

        self.s = s
        self.f = f
        self.fx = fx
        self.xr = xr

        return self

    def init(self):
        '''
        Set all main variables to default state
        (remove result of previous calculation).
        '''

        # Current values
        self.i = 0
        self.t = self.t_min
        self.x = self.x0.copy()
        self.r = None
        self.w = 0.

        # Values for each time step
        self.T = np.linspace(self.t_min, self.t_max, self.t_poi)
        self.X = []
        self.R = []
        self.W = []

        self.x_real = None # real (exact) solution
        self.x_calc = None # calculated solution
        self.x_err2 = None # x-calculation error

    def add(self, x, r):
        '''
        Add new step.
        '''

        self.i+= 1
        self.t = self.T[i]
        self.x = x.copy().reshape(self.d, self.m)
        self.r = r
        self.w+= np.sqrt(self.h) * np.random.randn((self.d, self.m))

        if True:
            self.X.append(self.x.copy())
            self.R.append(self.r.copy())
            self.W.append(self.w.copy())

    def check(self):
        '''
        Check calculation results.
        '''

        x_calc = self.x
        x_real = self.xr(self.t, self.w)

        print(x_calc)
        print(x_real)
        print(np.linalg.norm(x_real - x_calc, axis=0))

        return

        self.w = self.t



        self.x_real = None
        if self.xr is not None:
            self.x_real = np.array([
                self.xr(t_, w_) for t_, w_  in zip(self.T, self.W)
            ]).reshape(self.d, -1)

        self.x_err2 = None
        if self.x_real is not None and self.x_calc is not None:
            self.x_err2 = np.linalg.norm(self.x_real - self.x_calc, axis=0)

    def plot_x(self):
        '''
        Plot calculation result.
        '''

        if self.t is None or self.x_calc is None:
            return print('SDE is not ready. Can not plot')

        for i in range(self.x_calc.shape[0]):
            plt.plot(self.t, self.x_calc[i, :], label='x_%d'%(i+1))
        plt.title('SDE Solution')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend(loc='best')
        plt.show()

    def plot_x_err(self):
        '''
        Plot error of calculation result if exact solution is set.
        '''

        if self.t is None or self.x_err2 is None:
            return print('SDE is not ready. Can not plot')

        plt.plot(self.t, self.x_err2)
        plt.semilogy()
        plt.title('SDE Solution error')
        plt.xlabel('t')
        plt.ylabel('Error')
        plt.show()

    def plot_r(self):
        '''
        Plot distribution.
        '''

        if True:
            return print('SDE is not ready. Can not plot')
