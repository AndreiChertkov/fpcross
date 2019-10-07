import unittest
import numpy as np

from fpcross import polynomials_cheb

class TestPolynomial(unittest.TestCase):
    '''
    Tests for Polynomial class.
    '''

    def test_1(self):
        T = polynomials_cheb(X=0.5, m=4)
        self.assertEqual(T.shape, (5,))
        np.testing.assert_array_equal(T, np.array([ 1., 0.5, -0.5, -1., -0.5]))

if __name__ == '__main__':
    unittest.main()
