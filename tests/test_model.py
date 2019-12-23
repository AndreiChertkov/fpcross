import unittest
import numpy as np

from fpcross import Model


class TestModel(unittest.TestCase):
    '''
    Tests for the models.
    '''

    def test_1d_oup(self):
        MD = Model.select('fpe_1d_oup')
        self.assertEqual(MD.name(), 'fpe-1d-oup')
        MD = Model.select('fpe-1d-oup')
        self.assertEqual(MD.name(), 'fpe-1d-oup')
        self.assertEqual(MD._D, 0.5)
        self.assertEqual(MD._A, 1.)
        self.assertEqual(MD._s, 1.)

        MD.init(D=0.9, s=2.)
        self.assertEqual(MD._D, 0.9)
        self.assertEqual(MD._A, 1.)
        self.assertEqual(MD._s, 2.)

    def test_3d_oup(self):
        MD = Model.select('fpe_oup')
        self.assertEqual(MD.name(), 'fpe-oup')
        self.assertEqual(MD.d(), 1)
        self.assertEqual(MD._D, 0.5)
        np.testing.assert_array_equal(MD._A, np.array([[1.]]))
        self.assertEqual(MD._s, 1.)

        MD.init(d=3, D=0.9, s=2.)
        self.assertEqual(MD._d, 3)
        self.assertEqual(MD._D, 0.9)
        np.testing.assert_array_equal(MD._A, np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]))
        self.assertEqual(MD._s, 2.)

        A = np.array([
            [1., 0., 3.],
            [0., 2., 0.],
            [4., 0., 1.],
        ])
        MD.init(d=3, D=0.9, s=2., A=A)
        self.assertEqual(MD.d(), 3)
        self.assertEqual(MD._D, 0.9)
        np.testing.assert_array_equal(MD._A, A)
        self.assertEqual(MD._s, 2.)


if __name__ == '__main__':
    unittest.main()
