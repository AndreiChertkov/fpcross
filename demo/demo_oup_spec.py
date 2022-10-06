"""Demo script that shows how to use the Fokker-Planck solver for OUP."""
import numpy as np


from fpcross import EquationOUP
from fpcross import EquationOUPUniv
from fpcross import FPCross
from fpcross import FPCrossOUPDiag


def run_new():
    eq = EquationOUPUniv(d=3)
    eq.set_coef_rhs(np.array([
        [1.5, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.3, 1.0],
    ]))
    eq.init()

    fpc = FPCrossOUPDiag(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_3d_test_new')


def run_old():
    eq = EquationOUP(d=3)
    eq.set_coef_rhs(np.array([
        [1.5, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.3, 1.0],
    ]))
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_3d_test_old')


if __name__ == '__main__':
    run_old()
    run_new()
