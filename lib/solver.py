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

        x = self.EQ.x0.reshape(-1, 1)
        for t in self.EQ.t[:-1]:
            x_curr = self.step_x(x[:, -1], t)
            x = np.hstack([x, x_curr.reshape(-1, 1)])

        self.EQ.set_sol(x)

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

        f = self.EQ.f
        if callable(f): f = f(x, t)

        s = self.EQ.s
        if callable(s): s = s(x, t)

        e = np.random.randn(*x.shape)

        res = x + h * f + np.sqrt(h) * s.dot(e)

        return res

    def sde_step_r(self, r, x, t, fx, s, h):
        '''
        One step for probability distribution
        r = E r_0 (1-h*Tr(df/dx)).
        '''

        res = r * (1. - h * np.trace(fx_func(x, t)))

        return res
