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

class SolverTmp(Solver):

    def step(self, X):

        def step_x(X):

            def func_x(t, v):
                x = v.reshape(self.d, -1)
                f0 = self.func_f0(x, t)
                res = f0
                return res

            X0 = X.copy()
            for j in range(X.shape[1]):
                y1 = X[:, j]
                y2 = solve_ivp(func_x, [self.t, self.t - self.h], y1).y[:, -1]
                X0[:, j] = y2

            return X0

        def step_i(X):

            r0 = self.IT0.calc(X)
            return r0

        def step_w(X, w0):

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

            w = w0.copy()
            for j in range(X.shape[1]):
                if w0[j] < 1.E-30: # Zero
                    continue
                y1 = [*list(X[:, j]), np.log(w0[j])]
                y2 = solve_ivp(func_r, [self.t-self.h, self.t], y1).y[:, -1]
                w[j] = np.exp(y2[-1])

            return w

        def step_v(X, v0):
            v = self.Z @ v0

            return v

        X0 = step_x(X)
        r0 = step_i(X0)
        w1 = step_w(X0, r0)
        v1 = step_v(X0, w1)
        r1 = v1.copy()
        v1 = step_v(X0, r0)
        w1 = step_w(X0, v1)
        r2 = w1.copy()

        return (r1 + r2) / 2.
