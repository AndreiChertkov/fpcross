import unittest
import numpy as np

from fpcross import Grid


class TestGrid1d(unittest.TestCase):
    '''
    Tests for Grid class for the 1d case.
    '''

    def test_init1(self):
        n = 3
        l = [-2, 4]
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_init2(self):
        n = [3]
        l = np.array([-2, 4])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_init3(self):
        n = 3
        l = np.array([[-2, 4]])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_init4(self):
        n = np.array([3])
        l = np.array([[-2, 4]])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_init5(self):
        d = 1
        n = np.array([3])
        l = np.array([[-2, 4]])
        GR = Grid(d=d, n=n, l=l)

        self.assertEqual(GR.d, 1)
        np.testing.assert_array_equal(GR.n, np.array([3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.]]))

    def test_init6(self):
        d = -2
        n = np.array([3])
        l = np.array([[-2, 4]])

        msg = 'Invalid number of dimensions (d).'
        with self.assertRaises(ValueError) as ctx:
            GR = Grid(d=d, n=n, l=l)
        self.assertIn(msg, str(ctx.exception))

    def test_init7(self):
        d = 1
        n = [3, 4]
        l = np.array([[-2, 4]])

        msg = 'Invalid dimension for number of points (n).'
        with self.assertRaises(IndexError) as ctx:
            GR = Grid(d=d, n=n, l=l)
        self.assertIn(msg, str(ctx.exception))

    def test_pars(self):
        d = 1
        n = [11]
        l = [[-4., 6.]]
        k = 'u'
        e = 1.E-10

        GR = Grid(d, n, l, k, e)

        np.testing.assert_equal(GR.d, d)
        np.testing.assert_array_equal(GR.n, np.array(n))
        np.testing.assert_array_equal(GR.l, np.array(l))
        self.assertEqual(GR.k, k)
        self.assertEqual(GR.e, e)
        np.testing.assert_array_equal(GR.h, [1.])
        self.assertEqual(GR.n0, +11)
        self.assertEqual(GR.l1, -4.)
        self.assertEqual(GR.l2, +6.)
        self.assertEqual(GR.h0, +1.)

    def test_is_sym(self):
        self.assertTrue(Grid(n=5, l=[[-3., 3.]], k='u').is_sym())
        self.assertFalse(Grid(n=5, l=[[-4., 3.]], k='u').is_sym())
        self.assertFalse(Grid(n=5, l=[[1., 3.]], k='u').is_sym())


class TestGrid2d(unittest.TestCase):
    '''
    Tests for Grid class for the 2d case.
    '''

    def test_init1(self):
        d = 2
        n = 3
        l = [-2, 4]
        GR = Grid(d=d, n=n, l=l)

        self.assertEqual(GR.d, 2)
        np.testing.assert_array_equal(GR.n, np.array([3, 3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.], [-2., 4.]]))

    def test_init2(self):
        n = [3, 4]
        l = np.array([-2, 4])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 2)
        np.testing.assert_array_equal(GR.n, np.array([3, 4]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.], [-2., 4.]]))

    def test_init3(self):
        n = 3
        l = np.array([[-2, 4], [-5, 7]])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 2)
        np.testing.assert_array_equal(GR.n, np.array([3, 3]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.], [-5., 7.]]))

    def test_init4(self):
        n = np.array([3, 4])
        l = np.array([[-2, 4], [-5, 7]])
        GR = Grid(n=n, l=l)

        self.assertEqual(GR.d, 2)
        np.testing.assert_array_equal(GR.n, np.array([3, 4]))
        np.testing.assert_almost_equal(GR.l, np.array([[-2., 4.], [-5., 7.]]))

    def test_init5(self):
        d = 2
        n = np.array([3])
        l = np.array([[-2, 4]])

        msg = 'Invalid dimension for number of points (n).'
        with self.assertRaises(IndexError) as ctx:
            GR = Grid(d=d, n=n, l=l)
        self.assertIn(msg, str(ctx.exception))

    def test_init_kind(self):
        d = 2
        n = np.array([3, 4])
        l = np.array([[-2, 4], [-5, 7]])

        GR = Grid(d, n, l, 'u')
        self.assertEqual(GR.k, 'u')

        GR = Grid(d, n, l, 'c')
        self.assertEqual(GR.k, 'c')

        self.assertRaises(ValueError, lambda: Grid(d, n, l, 'xxx'))

    def test_pars(self):
        d = 2
        n = [11, 5]
        l = [[-4., 6.], [2., 4.]]
        k = 'u'
        e = 1.E-10

        GR = Grid(d, n, l, k, e)

        self.assertEqual(GR.d, d)
        np.testing.assert_array_equal(GR.n, n)
        np.testing.assert_array_equal(GR.l, np.array(l))
        self.assertEqual(GR.k, k)
        self.assertEqual(GR.e, e)
        np.testing.assert_array_equal(GR.h, [1., 0.5])
        np.testing.assert_array_equal(GR.n0, 8)
        self.assertEqual(GR.l1, -1.)
        self.assertEqual(GR.l2, +5.)
        self.assertEqual(GR.h0, +0.75)

    def test_copy(self):
        d = 2
        n1 = np.array([3, 4])
        l1 = np.array([[-2, 4], [-5, 7]])
        n2 = [4, 5]
        l2 = np.array([[-3, 5], [-6, 8]])

        GR1 = Grid(d, n1, l1, 'u')
        GR2 = GR1.copy(n=n2, l=l2)

        np.testing.assert_array_equal(GR1.n, n1)
        np.testing.assert_array_equal(GR2.n, n2)

        np.testing.assert_array_equal(GR1.l, l1)
        np.testing.assert_array_equal(GR2.l, l2)

    def test_copy_err(self):
        d = 2
        n1 = np.array([3, 4])
        l1 = np.array([[-2, 4], [-5, 7]])

        GR1 = Grid(d, n1, l1, 'u')

        msg = 'Invalid dimension for number of points (n).'
        with self.assertRaises(IndexError) as ctx:
            GR2 = GR1.copy(d=1)
        self.assertIn(msg, str(ctx.exception))

    def test_comp_u(self):
        GR = Grid(n=[5, 7], l=[
            [-4., 3.],
            [-1., 2.],
        ], k='u')

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
        ], k='c')

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
        np.testing.assert_almost_equal(GR.l1, -3.)
        np.testing.assert_almost_equal(GR.l2, +4.)
        np.testing.assert_equal(GR.h0, 0.5 * (7./7. + 7./9.))

    def test_square(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]])
        self.assertFalse(GR.is_square())

        GR = Grid(n=[5, 5], l=[[-4., 2.], [-4., 2.]])
        self.assertTrue(GR.is_square())

        GR = Grid(d=5, n=6, l=[2., 8.])
        self.assertTrue(GR.is_square())

    def test_find_u(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]], k='u')

        self.assertEqual(GR.find([-0.9, -1.2]), 2)
        self.assertEqual(GR.find([+1.9, -0.1]), 13)
        self.assertEqual(GR.find([11.9, 22.1]), 34)

    def test_find_c(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]], k='c')

        self.assertEqual(GR.find([-0.9, -1.2]), 32)
        self.assertEqual(GR.find([+1.9, -0.1]), 21)
        self.assertEqual(GR.find([11.9, 22.1]), 0)

    def test_is_out_u(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]], k='u')

        self.assertFalse(GR.is_out([-1., +1.4]))
        self.assertTrue(GR.is_out([+4., +1.4]))

    def test_is_out_c(self):
        GR = Grid(n=[5, 7], l=[[-4., 3.], [-1., 2.]], k='u')

        self.assertFalse(GR.is_out([-1., +1.4]))
        self.assertTrue(GR.is_out([+4., +1.4]))

    def test_is_sym(self):
        self.assertTrue(Grid(n=5, l=[[-3., 3.], [-4., 4.]], k='u').is_sym())
        self.assertFalse(Grid(n=5, l=[[-4., 3.], [-2, 2.]], k='u').is_sym())
        self.assertFalse(Grid(n=5, l=[[-2, 2.], [1., 3.]], k='u').is_sym())


class TestGrid3d(unittest.TestCase):
    '''
    Tests for Grid class for the 3d case.
    '''

    def test_init(self):
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

    def test_pars(self):
        d = 3
        n = [8, 7, 11]
        l = [[-4., 3.], [-1., 2.], [-5., 5.]]
        k = 'u'
        e = 1.E-10

        GR = Grid(d, n, l, k, e)

        self.assertEqual(GR.d, d)
        np.testing.assert_array_equal(GR.n, n)
        np.testing.assert_array_equal(GR.l, np.array(l))
        self.assertEqual(GR.k, k)
        self.assertEqual(GR.e, e)
        np.testing.assert_array_equal(GR.h, [1., 0.5, 1. ])
        self.assertEqual(GR.n0, 9)
        self.assertEqual(GR.l1, -3.3333333333333335)
        self.assertEqual(GR.l2, +3.3333333333333335)
        self.assertEqual(GR.h0, +2.5 / 3)

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


if __name__ == '__main__':
    unittest.main()
