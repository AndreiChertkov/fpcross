import unittest
import numpy as np

from fpcross import Grid, Func


def func_1d(x):
    return 2.*np.sin(np.pi * x[0, ])


def func_2d(x):
    return 2.*np.sin(np.pi * x[0, ]) + np.exp(-x[1, ])


def func_3d(x):
    return 2.*np.sin(np.pi * x[0, ]) + np.exp(-x[1, ]) + x[2, ]/2.


def func_4d(x):
    return 2.*np.sin(np.pi * x[0, ]) + np.exp(-x[1, ]) + x[2, ]/2. * x[3, ]**2


def func_for_int(X, d, a=2.):
    r = np.exp(-np.sum(X*X, axis=0) / a) / (np.pi * a)**(d/2)
    return r.reshape(-1)


class TestFunc(unittest.TestCase):
    '''
    Tests for Func class.
    '''

    def test_1d_np(self):
        GR = Grid(d=1, n=[50], l=[[-3., 4.]])
        FN = Func(GR)
        FN.init(func_1d)
        FN.prep()
        FN.calc()

        self.assertTrue(np.max(FN.test(1000, is_u=True)) < 1.0E-12)

    def test_1d_tt(self):
        GR = Grid(d=1, n=[50], l=[[-3., 4.]])
        FN = Func(GR, with_tt=True)
        FN.init(func_1d)
        FN.prep()
        FN.calc()

        self.assertTrue(np.max(FN.test(1000, is_u=True)) < 1.1E-12)

    def test_2d_np(self):
        GR = Grid(d=2, n=[28, 30], l=[
            [-3., 4.],
            [-2., 3.],
        ])
        FN = Func(GR)
        FN.init(func_2d)
        FN.prep()
        FN.calc()

        self.assertTrue(np.max(FN.test(1000, is_u=True)) < 8.0E-2)

    def test_2d_tt(self):
        GR = Grid(d=2, n=[28, 30], l=[
            [-3., 4.],
            [-2., 3.],
        ])
        FN = Func(GR, with_tt=True)
        FN.init(func_2d)
        FN.prep()
        FN.calc()

        self.assertTrue(np.max(FN.test(1000, is_u=True)) < 8.0E-2)

    def test_3d_np(self):
        GR = Grid(d=3, n=[24, 26, 28], l=[
            [-3., 4.],
            [-2., 3.],
            [-1., 1.],
        ])
        FN = Func(GR)
        FN.init(func_3d)
        FN.prep()
        FN.calc()

        self.assertTrue(np.max(FN.test(1000, is_u=True)) < 5.0E-2)

    def test_3d_tt(self):
        GR = Grid(d=3, n=[24, 26, 28], l=[
            [-3., 4.],
            [-2., 3.],
            [-1., 1.],
        ])
        FN = Func(GR, with_tt=True)
        FN.init(func_3d)
        FN.prep()
        FN.calc()

        self.assertTrue(np.max(FN.test(1000, is_u=True)) < 5.0E-2)

    def test_4d_np(self):
        GR = Grid(d=4, n=[30, 31, 32, 33], l=[
            [-3., 4.],
            [-2., 3.],
            [-1., 1.],
            [-1., 3.],
        ])
        FN = Func(GR)
        FN.init(func_4d)
        FN.prep()
        FN.calc()

        self.assertTrue(np.max(FN.test(1000, is_u=True)) < 3.0E-10)

    def test_4d_tt(self):
        GR = Grid(d=4, n=[30, 31, 32, 33], l=[
            [-3., 4.],
            [-2., 3.],
            [-1., 1.],
            [-1., 3.],
        ])
        FN = Func(GR, with_tt=True)
        FN.init(func_4d)
        FN.prep()
        FN.calc()

        self.assertTrue(np.max(FN.test(1000, is_u=True)) < 3.0E-10)

    def test_int_1d_np(self):
        GR = Grid(n=61, l=[-10., 10.])
        FN = Func(GR).init(lambda X: func_for_int(X, 1))
        FN.prep()
        FN.calc()

        e = np.abs(1. - FN.comp_int()) / np.abs(1.)
        self.assertTrue(e < 5.5E-13)

    def test_int_2d_np(self):
        GR = Grid(n=[60, 80], l=[-10., 10.])
        FN = Func(GR).init(lambda X: func_for_int(X, 2))
        FN.prep()
        FN.calc()

        e = np.abs(1. - FN.comp_int()) / np.abs(1.)
        self.assertTrue(e < 5.1E-13)

    def test_int_3d_np(self):
        GR = Grid(n=[35, 35, 35], l=[-10., 10.])
        FN = Func(GR).init(lambda X: func_for_int(X, 3))
        FN.prep()
        FN.calc()

        e = np.abs(1. - FN.comp_int()) / np.abs(1.)
        self.assertTrue(e < 6.0E-7)

    def test_int_3d_tt(self):
        return # DRAFT
        GR = Grid(n=[35, 35, 35], l=[-10., 10.])
        FN = Func(GR, eps=1.E-6, with_tt=True)
        FN.init(lambda X: func_for_int(X, 3))
        FN.prep()
        FN.calc()

        e = np.abs(1. - FN.comp_int()) / np.abs(1.)
        self.assertTrue(e < 6.0E-7)


if __name__ == '__main__':
    unittest.main()
