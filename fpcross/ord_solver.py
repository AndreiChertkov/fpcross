import numpy as np
from scipy.integrate import solve_ivp as sp_solve_ivp


class OrdSolver(object):
    '''
    Class for solution of system of ordinary differential equations (ODE)
    d y / d t = f(y, t)
    with multiple initial conditions by various methods.

    Basic usage:
    1 Initialize class instance with time grid and solver options.
    2 Call "init" with the right hand side.
    3 Call "comp" to obtain the solution for given initial condition.
    '''

    def __init__(self, TG, kind='rk4'):
        '''
        Init solver parameters.

        INPUT:

        TG - one-dimensional time grid
        type: fpcross.Grid
        * Only uniform grids are supported in the current version.

        kind - type of the solver.
        type: str
        enum:
            - 'eul' - the 1th order Euler solver
            - 'rk4' - the 4th order Runge-Kutta method.
            - 'ivp' - standard scipy solver
        '''

        self.TG = TG
        self.kind = kind

        if self.TG.d != 1 or self.TG.kind != 'u':
            raise ValueError('Invalid time grid (should be 1-dim. uniform).')

        if self.kind != 'eul' and self.kind != 'rk4' and self.kind != 'ivp':
            raise ValueError('Invalid solver type.')

    def init(self, f, with_y0=False):
        '''
        Init the main parameters of the class instance.

        INPUT:

        f - function that calculate the rhs of the ODE for given points
        type: function
            inp:
                y - values of the spatial variable
                type: ndarray [dimensions, number of points] of float
                t - value of the time
                type: float
            out:
                f - function values on y points for time t
                type: ndarray [dimensions, number of points] of float

        with_y0 - flag:
            True  - the 3th argument y0 for the current points will be passed
                    to the function f
            False - without y0 argument
        type: bool
        '''

        self.f = f
        self.with_y0 = with_y0

    def comp(self, y0):
        '''
        Compute solution of the ODE for the given initial condition.

        INPUT:

        y0 - initial values of the variable
        type: ndarray (or list) [dimensions, number of points] of float

        OUTPUT:

        y - solution at the final time step
        type: ndarray [dimensions, number of points] of float
        '''

        if not isinstance(y0, np.ndarray):
            y0 = np.array(y0)

        t = self.TG.l0[0]
        y = y0.copy()

        if self.kind == 'ivp':
            for j in range(y.shape[1]):
                def func(t, y):
                    y_ = y.reshape(-1, 1)
                    if not self.with_y0: return self.f(y_, t).reshape(-1)
                    return self.f(y_, t, y0[:, j].reshape(-1, 1)).reshape(-1)

                y[:, j] = sp_solve_ivp(
                    func, [self.TG.l0[0], self.TG.l0[1]], y[:, j]
                ).y[:, -1]

            return y

        def func(y, t):
            return self.f(y, t, y0) if self.with_y0 else self.f(y, t)

        for _ in range(1, self.TG.n0):
            if self.kind == 'eul':
                y+= self.TG.h0 * func(y, t)
            if self.kind == 'rk4':
                k1 = self.TG.h0 * func(y, t)
                k2 = self.TG.h0 * func(y + 0.5 * k1, t + 0.5 * self.TG.h0)
                k3 = self.TG.h0 * func(y + 0.5 * k2, t + 0.5 * self.TG.h0)
                k4 = self.TG.h0 * func(y + k3, t + self.TG.h0)
                y+= (k1 + k2 + k2 + k3 + k3 + k4) / 6.

            t+= self.TG.h0

        return y
