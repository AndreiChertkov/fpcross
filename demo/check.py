import numpy as np
import os
from time import perf_counter as tpc


from fpcross import EquationDif
from fpcross import EquationDum
from fpcross import EquationOUP
from fpcross import FPCross


def run():
    print(f'>>>>>  DIF-3D')
    t = tpc()
    run_dif_3d()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-3D')
    t = tpc()
    run_oup_3d()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-5D')
    t = tpc()
    run_oup_5d()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')


def run_np():
    print(f'>>>>>  DIF-3D (Numpy format)')
    t = tpc()
    run_dif_3d_np()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-1D (Numpy format)')
    t = tpc()
    run_oup_1d_np()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-2D (Numpy format)')
    t = tpc()
    run_oup_2d_np()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-3D (Numpy format)')
    t = tpc()
    run_oup_3d_np()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')


def run_spec():
    print(f'>>>>>  DUM-3D')
    t = tpc()
    run_dum_3d()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')


def run_dif_3d():
    eq = EquationDif(d=3)
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/dif_3d')


def run_dif_3d_np():
    eq = EquationDif(d=3, is_full=True)
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/dif_3d_np')


def run_dum_3d():
    eq = EquationDum(d=3)
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/dum_3d', is_spec=True)


def run_oup_1d_np():
    eq = EquationOUP(d=1, is_full=True)
    eq.set_coef_rhs(np.array([
        [1.]
    ]))
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_1d_np')


def run_oup_2d_np():
    eq = EquationOUP(d=2, is_full=True)
    eq.set_coef_rhs(np.array([
        [1.0, 0.5],
        [0.0, 1.0],
    ]))
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_2d_np')


def run_oup_3d():
    eq = EquationOUP(d=3)
    eq.set_coef_rhs(np.array([
        [1.5, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.3, 1.0],
    ]))
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_3d')


def run_oup_3d_np():
    eq = EquationOUP(d=3, is_full=True)
    eq.set_coef_rhs(np.array([
        [1.5, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.3, 1.0],
    ]))
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_3d_np')


def run_oup_5d():
    eq = EquationOUP(d=5)
    eq.set_coef_rhs(np.array([
        [1.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.5, 0.3, 0.2, 0.0, 1.0],
    ]))
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_5d')


if __name__ == '__main__':
    os.makedirs('./demo/result', exist_ok=True)

    run()
    run_np()
    run_spec()
