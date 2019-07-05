import numpy as np

class Solver(object):
    '''
    Class that represents solver for stochastic differential equations
    dx = f(t, x) dt + s(t, x) dW, fx = df / dx,
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
            x = self.step_x(self.EQ.t, self.EQ.x)
            r = None
            self.EQ.step(x, r)

    def step_x(self, t, x):
        '''
        One step for x by Euler-Mayruyama scheme
        x = x_0 + h f(x) + h**0.5 s eta
        Input:
            t - [float] current time
            x - [d or d x m nd.array] current value
        Output:
              - [d or d x m nd.array] new value
        '''

        h = self.EQ.h
        f = self.EQ.f(t, x)
        s = self.EQ.s(t, x)
        e = np.random.randn(*x.shape)
        x = x + h * f + np.sqrt(h) * s@e

        return x

    def step_r(self, t, x, r):
        '''
        One step for probability distribution by original scheme
        r = E r_0 (1-h*Tr(df/dx)).
        '''

        h = self.EQ.h
        fx = self.EQ.fx(x, t)
        r = r * (1. - h * np.trace(fx_func(x, t)))

        return r
