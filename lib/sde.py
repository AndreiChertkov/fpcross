import numpy as np
import matplotlib.pyplot as plt

class Sde(object):
    '''
    Class that represent stochastic differential equation.
    '''

    def __init__(self):
        self.clean()

    def clean(self):
        '''
        Set all main variables to default state
        (remove result of previous calculation).
        '''

        self.x_real = None # real (exact) solution
        self.x_calc = None # calculated solution
        self.x_err2 = None # x-calculation error

    def set(self, d, q, s, f, fx, x0, xr=None):
        '''
        Set equation parameters:
        dx = f(x, t) dt + s(x, t) dW, fx = df / dx,
        where x is a d-dim vector, dW is a q-dim noise,
        x0 is initial condition and xr is (optional) real solution.
        '''

        self.d = d
        self.q = q
        self.s = s
        self.f = f
        self.fx = fx
        self.x0 = x0.reshape(d, 1)
        self.xr = xr

        return self

    def set_desc(self, t_min=0., t_max=1., t_poi=100):
        '''
        Set descritization parameters.
        '''

        self.t_min = t_min
        self.t_max = t_max
        self.t_poi = t_poi
        self.t = np.linspace(t_min, t_max, t_poi)
        self.h = (t_max - t_min) / (t_poi - 1)

        return self

    def set_sol(self, x_calc=None):
        '''
        Set calculation result.
        '''

        self.x_calc = x_calc

        return self

    def prep(self):
        '''
        Prepare main parameters.
        '''

        self.w = self.t

        self.x_real = None
        if self.xr is not None:
            self.x_real = np.array([
                self.xr(t_, w_) for t_, w_  in zip(self.t, self.w)
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
