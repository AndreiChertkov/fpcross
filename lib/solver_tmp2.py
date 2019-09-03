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
from utils import rk4

class SolverTmp2(Solver):

    def step(self, X):

        def step_x(X):

            X0 = rk4(self.func_f0, X, self.t, self.t - self.h, t_poi=2)
            return X0

        def step_i(X):

            r0 = self.IT0.calc(X)
            return r0

        def step_w(X, w0):

            def func(v, t):
                x = v[:-1, :]
                r = v[-1, :]

                f0 = self.func_f0(x, t)
                f1 = self.func_f1(x, t)

                return np.vstack([f0, -np.trace(f1) * r])

            v0 = np.vstack([X, w0])
            v1 = rk4(func, v0, self.t - self.h, self.t, t_poi=2)
            w = v1[-1, :]
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
