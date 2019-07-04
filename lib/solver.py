import numpy as np
import matplotlib.pyplot as plt

class Solver(object):
    '''
    Class that represent solver for stochastic differential equations.
    '''

    def __init__(self, EQ=None):
        '''
        EQ - is a stochastic differential equation to solve
        '''

        self.EQ = EQ

    def step_x(self, x, t, f, s, h):
        '''
        One step of Euler-Mayruyama scheme
        Input:
            x - current value
                (1D ndarray)
            t - current time
                (float)
            f - function f(x, t) that return rhs or its value
                (1D ndarray or function)
            s - function s(x, t) that return stochastic part or its value
                (2D ndarray or function)
            h - time step
                (float)
        Output:
              - current value
                (1D ndarray)
        '''

        if callable(f): f = f(x, t)
        if callable(s): s = s(x, t)

        e = np.random.randn(*x.shape)

        res = x + h * f + np.sqrt(h) * s * e

        return res

    def sde_step_r(self, r, x, t, fx, s, h):
        '''
        One step for probability distribution
        r = E r_0 (1-h*Tr(df/dx))
        '''

        res = r * (1. - h * np.trace(fx_func(x, t)))

        return res
