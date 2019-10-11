import unittest
import numpy as np

from fpcross import Grid

class TestGrid(unittest.TestCase):
    '''
    Tests for Grid class.
    '''

    def test_1d_init1(self):
        n = 3
        l = [-2, 4]
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_1d_init2(self):
        n = [3]
        l = np.array([-2, 4])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_1d_init3(self):
        n = 3
        l = np.array([[-2, 4]])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_1d_init4(self):
        n = np.array([3])
        l = np.array([[-2, 4]])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_2d_init1(self):
        d = 2
        n = 3
        l = [-2, 4]
        GR = Grid(d=d, n=n, l=l)

        self.assertEqual(GR.d, 2)
        np.testing.assert_array_equal(GR.n, np.array([3, 3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.], [-2., 4.]]))

    def test_2d_init2(self):
        n = [3, 4]
        l = np.array([-2, 4])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 2)
        np.testing.assert_array_equal(GR.n, np.array([3, 4]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.], [-2., 4.]]))

    def test_2d_init3(self):
        n = 3
        l = np.array([[-2, 4], [-5, 7]])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 2)
        np.testing.assert_array_equal(GR.n, np.array([3, 3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.], [-5., 7.]]))

    def test_2d_init4(self):
        n = np.array([3, 4])
        l = np.array([[-2, 4], [-5, 7]])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 2)
        np.testing.assert_array_equal(GR.n, np.array([3, 4]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.], [-5., 7.]]))

    def test_3d_init(self):
        d = 3
        n = np.array([5, 7, 8])
        l = np.array([[-2., 3.], [-4., 5.], [-6., 9.]])
        GR = Grid(d=d, n=n, l=l)

        self.assertEqual(
            GR.d, 3
        )
        np.testing.assert_array_equal(
            GR.n, np.array([5, 7, 8])
        )
        np.testing.assert_almost_equal(
            GR.l, np.array([[-2., 3.], [-4., 5.], [-6., 9.]])
        )

    def test_init_kind(self):
        d = 2
        n = np.array([3, 4])
        l = np.array([[-2, 4], [-5, 7]])

        GR = Grid(d, n, l, 'u')
        np.testing.assert_equal(GR.kind, 'u')

        GR = Grid(d, n, l, 'c')
        np.testing.assert_equal(GR.kind, 'c')

        self.assertRaises(ValueError, lambda: Grid(d, n, l, 'xxx'))

    def test_copy(self):
        d = 3
        n = np.array([5, 7, 8])
        l = np.array([[-2., 3.], [-4., 5.], [-6., 9.]])
        GR1 = Grid(d=d, n=n, l=l)

        GR2 = GR1.copy()
        GR2.d = 1
        GR2.n = np.array([5])
        GR2.l = np.array([[-2., 3.]])

        self.assertEqual(
            GR1.d, 3
        )
        np.testing.assert_array_equal(
            GR1.n, np.array([5, 7, 8])
        )
        np.testing.assert_almost_equal(
            GR1.l, np.array([[-2., 3.], [-4., 5.], [-6., 9.]])
        )

        self.assertEqual(
            GR2.d, 1
        )
        np.testing.assert_array_equal(
            GR2.n, np.array([5])
        )
        np.testing.assert_almost_equal(
            GR2.l, np.array([[-2., 3.]])
        )

    def test_comp_u(self):
        GR = Grid(n=[5, 7], l=[
            [-4., 3.],
            [-1., 2.],
        ], kind='u')

        X = GR.comp([0, 0])
        np.testing.assert_almost_equal(
            X, np.array([[-4.], [-1.]])
        )

        X = GR.comp([4, 6])
        np.testing.assert_almost_equal(
            X, np.array([[3.], [2.]])
        )

        X = GR.comp([
            [0, 4],
            [0, 6],
        ])
        np.testing.assert_almost_equal(
            X, np.array([
                [-4., 3.],
                [-1., 2.],
            ])
        )

    def test_comp_c(self):
        GR = Grid(n=[5, 7], l=[
            [-4., 3.],
            [-1., 2.],
        ], kind='c')

        X = GR.comp([0, 0])
        np.testing.assert_almost_equal(
            X, np.array([[3.], [2.]])
        )

        X = GR.comp([4, 6])
        np.testing.assert_almost_equal(
            X, np.array([[-4.], [-1.]])
        )

        X = GR.comp([
            [0, 4],
            [0, 6],
        ])
        np.testing.assert_almost_equal(
            X, np.array([
                [3., -4.],
                [2., -1.],
            ])
        )

    def test_info(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]])
        s = GR.info(is_ret=True)

        self.assertTrue('Kind             : Chebyshev' in s)
        self.assertTrue('Dimensions       : 2' in s)
        self.assertTrue('Poi 5   | Min -4.000 | Max 3.000  |' in s)
        self.assertTrue('Poi 7   | Min -1.000 | Max 2.000  |' in s)

    def test_mean(self):
        n = np.array([8, 10])
        l = np.array([
            [-4., 3.],
            [-2., 5.],
        ])
        GR = Grid(2, n, l)

        np.testing.assert_equal(GR.n0, 9)
        np.testing.assert_almost_equal(GR.l0, np.array([-3., 4.]))
        np.testing.assert_equal(GR.h0, 0.5 * (7./7. + 7./9.))

    def test_square(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]])
        self.assertFalse(GR.is_square())

        GR = Grid(n=[5, 5], l=[[-4., 2.], [-4., 2.]])
        self.assertTrue(GR.is_square())

        GR = Grid(d=5, n=6, l=[2., 8.])
        self.assertTrue(GR.is_square())

    def test_indm(self):
        GR = Grid(n=[5, 7, 8])

        np.testing.assert_equal(GR.indm(0), np.array([0, 0, 0]))
        np.testing.assert_equal(GR.indm(1), np.array([1, 0, 0]))
        np.testing.assert_equal(GR.indm(5), np.array([0, 1, 0]))
        np.testing.assert_equal(GR.indm(279), np.array([4, 6, 7]))

    def test_indf(self):
        GR = Grid(n=[5, 7, 8])

        self.assertEqual(GR.indf([0, 0, 0]), 0)
        self.assertEqual(GR.indf([1, 0, 0]), 1)
        self.assertEqual(GR.indf([0, 1, 0]), 5)
        self.assertEqual(GR.indf([4, 6, 7]), 279)

    def test_find_u(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]], kind='u')

        self.assertEqual(GR.find([-0.9, -1.2]), 2)
        self.assertEqual(GR.find([+1.9, -0.1]), 13)
        self.assertEqual(GR.find([11.9, 22.1]), 34)

    def test_find_c(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]], kind='c')

        self.assertEqual(GR.find([-0.9, -1.2]), 32)
        self.assertEqual(GR.find([+1.9, -0.1]), 21)
        self.assertEqual(GR.find([11.9, 22.1]), 0)

    def test_is_out_u(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]], kind='u')

        self.assertFalse(GR.is_out([-1., +1.4]))
        self.assertTrue(GR.is_out([+4., +1.4]))

    def test_is_out_c(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]], kind='u')

        self.assertFalse(GR.is_out([-1., +1.4]))
        self.assertTrue(GR.is_out([+4., +1.4]))

if __name__ == '__main__':
    unittest.main()
