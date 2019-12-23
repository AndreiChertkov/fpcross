import unittest
import numpy as np

from fpcross import Grid
from fpcross.mtr import difscheb, dif1cheb, dif2cheb


class TestMtrDifscheb(unittest.TestCase):
    '''
    Tests for functions for construction of the Chebyshev differential matrices.
    '''

    def test_1ord_elements(self):
        N = 5
        SG = Grid(d=1, n=N+1, l=[-1., 1.])

        X = SG.comp()[0, :]
        D = dif1cheb(SG)

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

        D1, D2 = difscheb(SG, 2)

        e = D1 @ D1 - D2
        self.assertTrue(np.max(np.abs(e)) < 1.E-13)


if __name__ == '__main__':
    unittest.main()
