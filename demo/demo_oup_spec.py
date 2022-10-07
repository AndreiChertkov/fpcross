import numpy as np


from fpcross import EquationOUP
from fpcross import EquationOUPUniv
from fpcross import FPCross
from fpcross import FPCrossOUPDiag


def run(d, A, m, t, is_new=True):
    Equation_ = EquationOUPUniv if is_new else EquationOUP
    FPCross_ = FPCrossOUPDiag if is_new else FPCross

    eq = Equation_(d)
    eq.set_grid_time(m, t)
    eq.set_coef_rhs(A)
    eq.init()

    fpc = FPCross_(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_3d_test_' + ('new' if is_new else 'old'))


if __name__ == '__main__':
    a = [1.5, 1.0, 0.5, 2.0, 1.5, 1.2, 2.2, 3.3, 1.1, 5.1]
    A = np.diag(a)  # RHS matrix
    d = A.shape[0]  # Dimension
    m = 200         # Number of time points
    t = 10.         # Final time

    run(d, A, m, t)                # Run new solver (for diag OUP)
    # run(d, A, m, t, is_new=False)  # Run old solver (for general FPE)

    # Run it as "python demo/demo_oup_spec.py"
