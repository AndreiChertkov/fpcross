import unittest
import numpy as np

from fpcross import Grid, Solver, Model

class TestSolver(unittest.TestCase):
    '''
    Tests for solver.
    '''

    def test_fpe_1d_oup(self):
        MD = Model.select('fpe_1d_oup')
        MD.init(s=1., D=0.5, A=1.)

        SL = Solver(
            TG=Grid(d=1, n=1000, l=[+0., +8.], kind='u'),
            SG=Grid(d=1, n=101, l=[-6., +6.], kind='c'),
            MD=MD
        )
        SL.init()
        SL.prep()
        SL.calc(with_print=False)
        self.assertTrue(SL.hst['E_real'][-1] < 9.29e-06)
        self.assertTrue(SL.hst['E_stat'][-1] < 9.34e-06)

    def test_fpe_3d_drift_zero_tt(self):
        return # TODO: Remove.
        MD = Model.select('fpe_3d_drift_zero')
        MD.init(s=1., D=0.5)

        SL = Solver(
            TG=Grid(d=1, n=10, l=[+0., +1.], kind='u'),
            SG=Grid(d=3, n=21, l=[-5., +5.], kind='c'),
            MD=MD, eps=1.E-2, with_tt=True
        )
        SL.init()
        SL.prep()
        SL.calc(with_print=False)
        self.assertTrue(SL.hst['E_real'][-1] < 4.48e-03)

if __name__ == '__main__':
    unittest.main()
