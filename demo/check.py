"""Package fpcross, module demo.check: usage of the solver for various problems.

Script, which demonstrates how to apply the Fokker-Planck solver based on the
cross approximation method for different model problems. Run it from the root
of the project as "clear && python demo/check.py". The output should looks like
this (see also the folder "demo/result" with auto-generated plots):
"
>>>>>  DIF-3D
Solve: 100step [00:06, 16.01step/s,  | t=2.50 | r=1.0 | e_t=1.51e-02]
-----  Solved > Time :   6.8367 sec

>>>>>  OUP-3D
Solve: 100step [00:10,  9.58step/s,  | t=5.00 | r=4.0 | e_s=2.19e-03]
-----  Solved > Time :  10.9529 sec

>>>>>  OUP-5D
Solve: 100step [00:46,  2.16step/s,  | t=5.00 | r=4.5 | e_s=3.30e-03]
-----  Solved > Time :  46.9134 sec

>>>>>  DIF-3D (Numpy format)
Solve: 100step [00:54,  1.83step/s,  | t=2.50 | e_t=1.51e-02]
-----  Solved > Time :  54.8423 sec

>>>>>  OUP-1D (Numpy format)
Solve: 100step [00:00, 420.77step/s,  | t=5.00 | e_s=4.00e-04 | e_t=4.00e-04]
-----  Solved > Time :   0.6074 sec

>>>>>  OUP-2D (Numpy format)
Solve: 100step [00:02, 35.30step/s,  | t=5.00 | e_s=7.68e-04]
-----  Solved > Time :   3.1674 sec

>>>>>  OUP-3D (Numpy format)
Solve: 100step [03:19,  1.99s/step,  | t=5.00 | e_s=1.35e-03]
-----  Solved > Time : 199.7536 sec

>>>>>  DUM-3D
Solve: 100step [01:43,  1.04s/step,  | t=10.00 | r=7.9 | e_eta=2.04e-03 | e_psi=2.72e-03]
-----  Solved > Time : 103.9188 sec
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
    _calc_dif_3d()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-3D')
    t = tpc()
    _calc_oup_3d()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-5D')
    t = tpc()
    _calc_oup_5d()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')


def run_np():
    print(f'>>>>>  DIF-3D (Numpy format)')
    t = tpc()
    _calc_dif_3d_np()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-1D (Numpy format)')
    t = tpc()
    _calc_oup_1d_np()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-2D (Numpy format)')
    t = tpc()
    _calc_oup_2d_np()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')

    print(f'>>>>>  OUP-3D (Numpy format)')
    t = tpc()
    _calc_oup_3d_np()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')


def run_spec():
    print(f'>>>>>  DUM-3D')
    t = tpc()
    _calc_dum_3d()
    print(f'-----  Solved > Time : {tpc()-t:-8.4f} sec\n')


def _calc_dif_3d():
    eq = EquationDif(d=3)
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/dif_3d')


def _calc_dif_3d_np():
    eq = EquationDif(d=3, is_full=True)
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/dif_3d_np')


def _calc_dum_3d():
    eq = EquationDum(d=3)
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/dum_3d', is_spec=True)


def _calc_oup_1d_np():
    eq = EquationOUP(d=1, is_full=True)
    eq.set_coef_rhs(np.array([
        [1.]
    ]))
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_1d_np')


def _calc_oup_2d_np():
    eq = EquationOUP(d=2, is_full=True)
    eq.set_coef_rhs(np.array([
        [1.0, 0.5],
        [0.0, 1.0],
    ]))
    eq.init()

    fpc = FPCross(eq, with_hist=True)
    fpc.solve()
    fpc.plot('./demo/result/oup_2d_np')


def _calc_oup_3d():
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


def _calc_oup_3d_np():
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


def _calc_oup_5d():
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
