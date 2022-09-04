"""Apply the Fokker-Planck solver for different model problems.

Run it from the root of the project as "clear && python demo/check.py". The
output should look like this (see also the folder "demo/result" with plots):
"
>>>>>  DIF-3D
Solve: 100step [00:06, 15.92step/s,  | t=2.50 | r=1.0 | e_t=1.50e-02]
-----  Solved > Time :   7.5590 sec

>>>>>  OUP-3D
Solve: 100step [00:13,  7.40step/s,  | t=5.00 | r=4.0 | e_s=1.57e-03]
-----  Solved > Time :  13.9383 sec

>>>>>  OUP-5D
Solve: 100step [01:07,  1.48step/s,  | t=5.00 | r=4.5 | e_s=1.94e-03]
-----  Solved > Time :  68.1668 sec

>>>>>  DIF-3D (Numpy format)
Solve: 100step [00:59,  1.68step/s,  | t=2.50 | e_t=1.50e-02]
-----  Solved > Time :  59.8291 sec

>>>>>  OUP-1D (Numpy format)
Solve: 100step [00:00, 645.54step/s,  | t=5.00 | e_s=3.59e-04 | e_t=3.59e-04]
-----  Solved > Time :   0.5112 sec

>>>>>  OUP-2D (Numpy format)
Solve: 100step [00:02, 34.89step/s,  | t=5.00 | e_s=7.07e-04]
-----  Solved > Time :   3.1149 sec

>>>>>  OUP-3D (Numpy format)
Solve: 100step [03:39,  2.20s/step,  | t=5.00 | e_s=1.98e-03]
-----  Solved > Time : 220.1667 sec

>>>>>  DUM-3D
Solve: 100step [02:13,  1.33s/step,  | t=10.00 | r=8.2 | e_eta=8.57e-04 | e_psi=1.47e-04]
-----  Solved > Time : 133.4268 sec
".

"""
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
