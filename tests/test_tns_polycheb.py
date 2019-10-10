import unittest
import numpy as np

from fpcross import polycheb

class TestTnsPolycheb(unittest.TestCase):
    '''
    Tests for Chebyshev polynomials function.
    '''

    def test_1d_simple(self):
        T = polycheb(X=0.5, m=5)
        self.assertEqual(T.shape, (5,))
        np.testing.assert_array_equal(T, np.array([ 1., 0.5, -0.5, -1., -0.5]))

    def test_1d_1p(self):
        T = polycheb(X=-3., m=5, l=[[-3., 3.]])
        self.assertEqual(T.shape, (5,))
        np.testing.assert_array_equal(T, np.array([ 1., -1.,  1., -1.,  1.]))

    def test_1d(self):
        T = polycheb(X=[-3., -3.], m=5, l=[[-3., 3.]])
        self.assertEqual(T.shape, (5, 2))
        np.testing.assert_array_equal(T, np.array([
            [ 1., -1.,  1., -1.,  1.],
            [ 1., -1.,  1., -1.,  1.],
        ]).T)

    def test_2d(self):
        T = polycheb(
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

if __name__ == '__main__':
    unittest.main()
