import numpy as np
import matplotlib.pyplot as plt

class Sde(object):
    '''
    Class that represent stochastic differential equation.
    '''

    def __init__(self):
        '''

        '''

        self.set()


    def set(self, d=None, q=None, s=None, f=None, fx=None, x0=None, xr=None):
        '''
        Set equation parameters:
        dx = f(x, t) dt + s(x, t) dW, fx = df / dx,
        where x is d-dim vector, dW is q-dim noise,
        x0 is initial condition and xr is (optional) real solution.
        '''

        self.d = d
        self.q = q
        self.s = s
        self.f = f
        self.fx = fx
        self.x0 = x0
        self.xr = xr

        return self

    def set_desc(self, t_min=0., t_max=1., t_poi=100):
        '''
        Set descritization parameters.
        '''

        self.t_min = t_min
        self.t_max = t_max
        self.t_poi = t_poi
        self.t_list = np.linspace(t_min, t_max, t_poi)
        self.h = (t_max - t_min) / (t_poi - 1)

        return self
