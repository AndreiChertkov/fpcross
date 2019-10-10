import unittest
import numpy as np

from fpcross import Grid, Func

def func_1d(x):
    return 2. * np.sin(np.pi * x[0, ])

def func_2d(x):
    return 2. * np.sin(np.pi * x[0, ]) + np.exp(-x[1, ])

def func_3d(x):
    return 2. * np.sin(np.pi * x[0, ]) + np.exp(-x[1, ]) + x[2, ] / 2.

def func_4d(x):
    return 2. * np.sin(np.pi * x[0, ]) + np.exp(-x[1, ]) + x[2, ] / 2. * x[3, ]**2

def func_5d(x):
    f = 2. * np.sin(np.pi * x[0, ])
    f+= np.exp(-x[1, ])
    f+= x[2, ] / 2. * x[3, ]**2
    f+= 3. * np.cos(np.pi * x[4, ]**2)
    return f

def func_10d(x):
    d = 10
    f = 0.
    for k in range(d):
        f+= np.sin(np.pi / (k+1.) * x[k, ]**2)
    return f

def func_99d(x):
    d = 10
    f = 0.
    for k in range(d):
        f+= np.sin(np.pi / (k+1.) * x[k, ]**2)
    return f

class TestFunc(unittest.TestCase):
    '''
    Tests for Func class.
    '''

    def test_1d_np(self):
        GR = Grid(d=1, n=[50], l=[[-3., 4.]])
        FN = Func(GR, with_tt=False)
        FN.init(func_1d)
        FN.prep()
        FN.calc()
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-10)

    def test_1d_tt(self):
        GR = Grid(d=1, n=[50], l=[[-3., 4.]])
        FN = Func(GR)
        FN.init(func_1d)
        FN.prep()
        FN.calc()
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-10)

    def test_2d_np(self):
        GR = Grid(d=2, n=[28, 30], l=[
            [-3., 4.],
            [-2., 3.],
        ])
        FN = Func(GR, with_tt=False)
        FN.init(func_2d)
        FN.prep()
        FN.calc()
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-3)

    def test_2d_tt(self):
        GR = Grid(d=2, n=[28, 30], l=[
            [-3., 4.],
            [-2., 3.],
        ])
        FN = Func(GR)
        FN.init(func_2d)
        FN.prep()
        FN.calc()
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-3)

    def test_3d_np(self):
        GR = Grid(d=3, n=[24, 26, 28], l=[
            [-3., 4.],
            [-2., 3.],
            [-1., 1.],
        ])
        FN = Func(GR, with_tt=False)
        FN.init(func_3d)
        FN.prep()
        FN.calc()
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-3)

    def test_3d_tt(self):
        GR = Grid(d=3, n=[24, 26, 28], l=[
            [-3., 4.],
            [-2., 3.],
            [-1., 1.],
        ])
        FN = Func(GR)
        FN.init(func_3d)
        FN.prep()
        FN.calc()
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-3)

    def test_4d_np(self):
        GR = Grid(d=4, n=[30, 31, 32, 33], l=[
            [-3., 4.],
            [-2., 3.],
            [-1., 1.],
            [-1., 3.],
        ])
        FN = Func(GR, with_tt=False)
        FN.init(func_4d)
        FN.prep()
        FN.calc()
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-7)

    def test_4d_tt(self):
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
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-7)

    def test_10d_tt(self):
        d = 10
        n = np.linspace(50, 70, d, dtype='int')
        l = []
        for k in range(d):
            l.append([-2. - k * 0.2, +2. + k * 0.2])

        GR = Grid(d, n, l)
        FN = Func(GR)
        FN.init(func_10d)
        FN.prep()
        FN.calc()
        err = FN.test(100)

        self.assertTrue(np.max(err) < 1.E-10)

if __name__ == '__main__':
    unittest.main()
