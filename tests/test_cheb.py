import unittest
import numpy as np

from fpcross import Grid
from fpcross.cheb import poly, difs, dif1


class TestPoly(unittest.TestCase):
    '''
    Tests for Chebyshev polynomials function.
    '''

    def test_1d_simple(self):
        T = poly(X=0.5, m=5)
        self.assertEqual(T.shape, (5,))
        np.testing.assert_array_equal(T, np.array([ 1., 0.5, -0.5, -1., -0.5]))

    def test_1d_1p(self):
        T = poly(X=-3., m=5, l=[[-3., 3.]])
        self.assertEqual(T.shape, (5,))
        np.testing.assert_array_equal(T, np.array([ 1., -1.,  1., -1.,  1.]))

    def test_1d(self):
        T = poly(X=[-3., -3.], m=5, l=[[-3., 3.]])
        self.assertEqual(T.shape, (5, 2))
        np.testing.assert_array_equal(T, np.array([
            [ 1., -1.,  1., -1.,  1.],
            [ 1., -1.,  1., -1.,  1.],
        ]).T)

    def test_2d(self):
        T = poly(
            X=[[-3., 0., +3., +3.], [3., 0., -3., +3.]],
            m=3,
            l=[[-3., 3.]]
        )
        self.assertEqual(T.shape, (3, 2, 4))
        np.testing.assert_array_equal(T[:, :, 0], np.array([
            [ 1., -1.,  1.],
            [ 1., +1.,  1.],
        ]).T)
        np.testing.assert_array_equal(T[:, :, 1], np.array([
            [ 1., 0.,  -1.],
            [ 1., 0.,  -1.],
        ]).T)
        np.testing.assert_array_equal(T[:, :, 2], np.array([
            [ 1., +1.,  1.],
            [ 1., -1.,  1.],
        ]).T)

        np.testing.assert_array_equal(T[:, :, 3], np.array([
            [ 1., +1.,  1.],
            [ 1., +1.,  1.],
        ]).T)


class TestDifs(unittest.TestCase):
    '''
    Tests for Chebyshev differential matrices functions.
    '''

    def test_1ord_elements(self):
        N = 5
        SG = Grid(d=1, n=N+1, l=[-1., 1.])

        X = SG.comp()[0, :]
        D = dif1(SG)

        e = D[0, 0] - (2.*N**2+1.) / 6.
        self.assertTrue(np.max(np.abs(e)) < 1.E-14)

        e = D[0, 2] - 2. / (1. - X[2])
        self.assertTrue(np.max(np.abs(e)) < 1.E-14)

        e = D[0, 3] + 2. / (1. - X[3])
        self.assertTrue(np.max(np.abs(e)) < 1.E-14)

        e = D[4, 4] + X[4] / (1. - X[4]**2) / 2.
        self.assertTrue(np.max(np.abs(e)) < 1.E-14)

    def test_2ord_elements(self):
        N = 5
        SG = Grid(d=1, n=N+1, l=[-1., 1.])

        D1, D2 = difs(SG, 2)

        e = D1 @ D1 - D2
        self.assertTrue(np.max(np.abs(e)) < 1.E-13)


if __name__ == '__main__':
    unittest.main()
