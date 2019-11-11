import numpy as np
from scipy.integrate import solve_ivp

class OrdSolver(object):
    '''
    Class for solution of system of ordinary differential equations (ODE)
    d y / d t = f(y, t)
    with multiple initial conditions (several different initial points)
    by various methods.

    Basic usage:
    1 Initialize class instance with time grid and solver options.
    2 Call "init" with the right hand side.
    3 Call "comp" to obtain the solution for given initial condition.
    '''

    def __init__(self, TG, kind='rk4', is_rev=False):
        '''
        INPUT:

        TG - one-dimensional time grid
        type: fpcross.Grid
        * Only uniform grids are supported in the current version.

        kind - type of the solver
        type: str
        enum:
            - 'eul' - the 1th order Euler solver
            - 'rk4' - the 4th order Runge-Kutta solver.
            - 'ivp' - standard scipy solver

        is_rev - flag:
            True  - equation will be solved from final to initial grid point
            False - equation will be solved from initial to final grid point
        type: bool
        '''

        self.TG = TG
        self.kind = str(kind)
        self.is_rev = bool(is_rev)

        if self.TG.d != 1 or self.TG.kind != 'u':
            raise ValueError('Invalid time grid (should be 1-dim. uniform).')
        if self.kind != 'eul' and self.kind != 'rk4' and self.kind != 'ivp':
            raise ValueError('Invalid solver type.')

        self.init()

    def init(self, f=None, with_y0=False):
        '''
        Init the main parameters of the class instance.

        INPUT:

        f - function that calculate the rhs of the ODE for given points
        type1: None
        type2: function
            inp:
                y - values of the spatial variable
                type: ndarray [dimensions, number of points] of float
                t - value of the time
                type: float
                y0 - (if with_y0) values of the related initial conditions
                type: ndarray [dimensions, number of points] of float
            out:
                v - function values on y points for time t
                type: ndarray [dimensions, number of points] of float

        with_y0 - flag:
            True  - the 3th argument y0 for the current points will be passed
                    to the function f
            False - without y0 argument
        type: bool

        OUTPUT:

        OS - self
        type: fpcross.OrdSolver
        '''

        self.f = f
        self.with_y0 = bool(with_y0)

        return self

    def comp(self, y0):
        '''
        Compute solution of the ODE for the given initial condition.

        INPUT:

        y0 - initial values of the variable
        type1: list [dimensions, number of points] of float
        type2: ndarray [dimensions, number of points] of float

        OUTPUT:

        y - solution at the final time step
        type: ndarray [dimensions, number of points] of float
        '''

        if not self.f:
            raise ValueError('Rhs for the equation is not set yet. Call "init" before.')

        if not isinstance(y0, np.ndarray):
            y0 = np.array(y0)

        t1 = self.TG.l1 if not self.is_rev else self.TG.l2
        t2 = self.TG.l2 if not self.is_rev else self.TG.l1
        h0 = self.TG.h0 if not self.is_rev else self.TG.h0 * -1.

        t = t1
        y = y0.copy()

        if self.kind == 'ivp':
            for j in range(y.shape[1]):
                def func(t, y):
                    y_ = y.reshape(-1, 1)
                    if self.with_y0:
                        v = self.f(y_, t, y0[:, j].reshape(-1, 1))
                    else:
                        v = self.f(y_, t)
                    return v.reshape(-1)

                y[:, j] = solve_ivp(func, [t1, t2], y[:, j]).y[:, -1]

            return y

        def func(y, t):
            return self.f(y, t, y0) if self.with_y0 else self.f(y, t)

        for _ in range(1, self.TG.n0):
            if self.kind == 'eul':
                y+= h0 * func(y, t)
            if self.kind == 'rk4':
                k1 = h0 * func(y, t)
                k2 = h0 * func(y + 0.5 * k1, t + 0.5 * h0)
                k3 = h0 * func(y + 0.5 * k2, t + 0.5 * h0)
                k4 = h0 * func(y + k3, t + h0)
                y+= (k1 + k2 + k2 + k3 + k3 + k4) / 6.

            t+= h0

        return y
