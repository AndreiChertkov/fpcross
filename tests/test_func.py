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

if __name__ == '__main__':
    unittest.main()
