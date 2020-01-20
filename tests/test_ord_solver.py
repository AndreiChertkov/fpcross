import unittest
import numpy as np

from fpcross import Grid, OrdSolver


def func_f0_1d(r, t, r0):
    return r*r


def func_rt_1d(t, r0):
    return r0 / (1. - r0 * t)


def func_f0_2d(r, t, r0):
    v1 = +4. * t * (r[1, :] - r0[1, :] + 1.)
    v2 = -4. * t * (r[0, :] - r0[0, :])
    return np.vstack([v1, v2])


def func_rt_2d(t, r0):
    v1 = np.sin(2. * t * t) + r0[0, :]
    v2 = np.cos(2. * t * t) + r0[1, :] - 1.
    return np.vstack([v1, v2])


class TestOrdSolver1d(unittest.TestCase):

    def test_1d_eul(self):
        TG = Grid(1, 2, [0., 0.0001], k='u')
        r0 = -2.5 * np.arange(1000).reshape(1, -1) + 0.2
        r_real = func_rt_1d(TG.l2, r0)

        SL = OrdSolver(TG, 'eul').init(func_f0_1d, with_y0=True)
        e = np.mean(np.abs((r_real - SL.comp(r0)) / r_real))

        self.assertTrue(e < 3.E-02)

    def test_1d_rk4(self):
        TG = Grid(1, 2, [0., 0.0001], k='u')
        r0 = -2.5 * np.arange(1000).reshape(1, -1) + 0.2
        r_real = func_rt_1d(TG.l2, r0)

        SL = OrdSolver(TG, 'rk4').init(func_f0_1d, with_y0=True)
        e = np.mean(np.abs((r_real - SL.comp(r0)) / r_real))

        self.assertTrue(e < 4.E-06)

    def test_1d_ivp(self):
        TG = Grid(1, 2, [0., 0.0001], k='u')
        r0 = -2.5 * np.arange(1000).reshape(1, -1) + 0.2
        r_real = func_rt_1d(TG.l2, r0)

        SL = OrdSolver(TG, 'ivp').init(func_f0_1d, with_y0=True)
        e = np.mean(np.abs((r_real - SL.comp(r0)) / r_real))

        self.assertTrue(e < 2.E-06)


class TestOrdSolver2d(unittest.TestCase):

    def test_2d_eul(self):
        TG = Grid(1, 2, [0., 0.001], k='u')
        r0 = np.vstack([
            np.arange(100) * 1.1 + 0.2,
            np.arange(100) * 1.5 + 0.3,
        ])
        r_real = func_rt_2d(TG.l2, r0)

        SL = OrdSolver(TG, 'eul').init(func_f0_2d, with_y0=True)
        e = np.mean(np.abs((r_real - SL.comp(r0)) / r_real))

        self.assertTrue(e < 1.E-07)


if __name__ == '__main__':
    unittest.main()
