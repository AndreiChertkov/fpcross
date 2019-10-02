import unittest
import numpy as np

class Test_divkgrad_1d_hd_analyt(unittest.TestCase):
    ''' Base class for 1D diffusion problem  '''

    def setUp(self):
        self.PDE = Pde()
        self.PDE.set_model('divkgrad_1d_hd_analyt')
        self.PDE.set_params([np.pi*2])
        self.PDE.set_tau(tau=1.E-14, eps_lss=1.E-14, tau_lss=1.E-14)
        self.PDE.set_lss_params(nswp=20, kickrank=4, local_prec='n', local_iters=2,
                                local_restart=20, trunc_norm=1, max_full_size=100)
        self.PDE.update_d(8)

class TestBase(unittest.TestCase):
    '''
    Tests for...
    '''

    def test_np(self):
        self.assertTrue(22  < 25)

    def test_tt(self):
        self.assertTrue(22  < 5.16E-05)

if __name__ == '__main__':
    unittest.main()
