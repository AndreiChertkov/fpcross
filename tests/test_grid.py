import unittest
import numpy as np

from fpcross import Grid

class TestGrid(unittest.TestCase):
    '''
    Tests for Grid class.
    '''

    def test_1d_short(self):
        n = 3
        l = [-2, 4]
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        self.assertEqual(GR.n, np.array([3.]))
        self.assertEqual(GR.l, np.array([[-2., 4.]]))

    def test_3d_full(self):
        d = 3
        n = np.array([5, 7, 8])
        l = np.array([[-2., 3.], [-4., 5.], [-6., 9.]])
        GR = Grid(d=d, n=n, l=l)

        self.assertEqual(GR.d, 3)

if __name__ == '__main__':
    unittest.main()
