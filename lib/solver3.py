import time

import numpy as np
from numpy import kron as kron
from scipy.linalg import expm as expm
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from tqdm import tqdm

from intertrain import Intertrain
from solver import Solver

class Solver3(Solver):

    def prep(self):
        '''
        Init calculation parameters, prepare interpolation of
        the initial condition and calculate special matrices.
        '''

        _t = []

        self._t_prep = None
        self._t_calc = None

        self.IT0 = None # Interpolant from the previous step
        self.IT1 = None # Initial interpolant
        self.IT2 = None # Final interpolant

        self.k = 0          # Current time step
        self.t = self.t_min # Current time value

        _t.append(time.time())

        self.IT.init(self.func_r0).prep()

        self.IT.dif2()
        self.D1 = self.IT.D1
        self.D2 = self.IT.D2

        if self.func_s0 is not None:
            h = (self.h) ** (1./self.d)
            J = np.eye(self.n); J[0, 0] = 0.; J[-1, -1] = 0.
            D = expm(h * J @ self.D2) @ J

            self.Z = D.copy()
            for d in range(self.d-1):
                self.Z = kron(self.Z, D)
        else:
            self.Z = np.eye(self.n**self.d)

        _t[-1] = time.time() - _t[-1]

        self._t_prep = sum(_t)

        self.X = self.IT.grid()
        self.R = []
        self.R.append(self.IT.calc(self.X))

        self.IT1 = self.IT.copy()

    def step(self, X):
        return self.step1(X) # (self.step1(X) + self.step2(X)) / 2.

    def step1(self, X):

        def func_x(t, v):
            x = v.reshape(self.d, -1)
            f0 = self.func_f0(x, t)
            res = f0
            return res

        def func_r(t, v):
            x = v[:-1].reshape(self.d, -1)
            r = v[-1]
            f0 = self.func_f0(x, t)
            f1 = self.func_f1(x, t)
            res = [
                *list(f0.reshape(-1)),
                -np.trace(f1)
            ]
            return res

        X0 = X.copy()
        for j in range(X.shape[1]):
            v1 = X[:, j]
            v2 = solve_ivp(func_x, [self.t, self.t - self.h], v1).y[:, -1]
            X0[:, j] = v2

        r0 = self.IT0.calc(X0)

        r1 = r0.copy()
        for j in range(X.shape[1]):
            if r1[j] < 1.E-18: continue
            v1 = [*list(X0[:, j]), np.log(r1[j])]
            v2 = solve_ivp(func_r, [self.t-self.h, self.t], v1).y[:, -1]
            r1[j] = np.exp(v2[-1])

        r2 = self.Z @ r1

        r = r2

        return r.reshape(-1)

    def step2(self, X):

        def func_x(t, v):
            x = v.reshape(self.d, -1)
            f0 = self.func_f0(x, t)
            res = f0
            return res

        def func_r(t, v):
            x = v[:-1].reshape(self.d, -1)
            r = v[-1]
            f0 = self.func_f0(x, t)
            f1 = self.func_f1(x, t)
            res = [
                *list(f0.reshape(-1)),
                -np.trace(f1) * r
            ]
            return res

        X0 = X.copy()
        for j in range(X.shape[1]):
            v1 = X[:, j]
            v2 = solve_ivp(func_x, [self.t, self.t - self.h], v1).y[:, -1]
            X0[:, j] = v2

        r0 = self.IT0.calc(X0)

        r1 = self.Z @ r0

        r2 = r0.copy()
        for j in range(X.shape[1]):
            v1 = [*list(X0[:, j]), r1[j]]
            v2 = solve_ivp(func_r, [self.t-self.h, self.t], v1).y[:, -1]
            r2[j] = v2[-1]

        r = r2

        return r.reshape(-1)
