import numpy as np
import matplotlib.pyplot as plt

class Solver(object):
    '''
    Class that represents solver for stochastic differential equations
    dx = f(x, t) dt + s(x, t) dW, fx = df / dx,
    where x is a d-dim vector, dW is a q-dim noise (Brownian motion),
    x0 is initial condition.
    '''

    def __init__(self, EQ):
        '''
        EQ - [instance of Equation] stochastic differential equation to solve
        '''

        self.EQ = EQ

    def solve_ode(self):
        '''
        Solve ODE (without stochastic part).
        '''

        self.EQ.init()

        for _ in range(self.EQ.t_poi - 1):
            x = self.step_x(self.EQ.x, self.EQ.t)
            r = None
            self.EQ.add(x, r)

    def step_x(self, x, t):
        '''
        One step of Euler-Mayruyama scheme
        Input:
            x - current value
                (1D ndarray)
            t - current time
                (float)
        Output:
              - new value
                (1D ndarray)
        '''

        h = self.EQ.h
        f = self.EQ.f(x, t)
        s = self.EQ.s(x, t)
        e = np.random.randn(*x.shape)
        return x + h * f + np.sqrt(h) * s.dot(e)

    def sde_step_r(self, r, x, t, fx, s, h):
        '''
        One step for probability distribution
        r = E r_0 (1-h*Tr(df/dx)).
        '''

        res = r * (1. - h * np.trace(fx_func(x, t)))

        return res
