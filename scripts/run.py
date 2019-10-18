# clear && clear && python setup.py install && python ./tests/test.py
# clear && clear && python setup.py install && python ./scripts/run.py

import numpy as np
from fpcross import Grid, Solver, Model, Check

def run():
    MD = Model.select('fpe_oup')
    MD.init(d=3, s=1., D=0.5, A=np.array([
        [1.0, 0.2, 0.5],
        [0.0, 0.7, 0.3],
        [0.0, 0.0, 1.5],
    ]))

    TG = Grid(d=1, n=11, l=[+0., +8.], kind='u')
    SG = Grid(d=3, n= 20, l=[-6., +6.], kind='c')
    SL = Solver(TG, SG, MD, with_tt=True, eps=1.E-6)
    SL.init().prep().calc().info()

    SL = Solver(TG, SG, MD)
    SL.init().prep().calc().info()

if __name__ == '__main__':
    print('\n'*5)
    run()
