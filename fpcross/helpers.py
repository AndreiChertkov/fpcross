import os
import time
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tt

css = '''
    div {
      margin: 0;
      padding: 0;
    }

    .head0 {
      padding: 10px;
      display: flex;
      background-color: #dedee2;
    }
    .head0__name {
      padding-right: 5px;
      flex: 1 1 50%;
      align-self: center;
      font-size: 36px;
      font-weight: bold;
      color: #6b7e51;
    }
    .head0__note {
      padding-left: 10px;
      flex: 1 1 50%;
      align-self: flex-end;
      font-size: 16px;
      font-style: italic;
      color: #485536;
      border-left: 5px solid #8b1d1d;
    }

    .head1 {
      margin-top: 30px;
      padding: 10px;
      border-top: 5px solid #8b1d1d;
    }
    .head1__name {
      font-size: 32px;
      font-weight: bold;
      color: #485536;
    }
    .head1__note {
      padding-left: 20px;
      font-size: 20px;
      font-style: italic;
      color: #485536;
    }

    .head2 {
      margin-top: 7px;
      padding: 5px 30px 5px;
      border-left: 2px solid #42d56e;
    }
    .head2__name {
      display: inline-block;
      max-width: 70%;
      padding: 2px 10px;
      font-size: 22px;
      font-weight: bold;
      color: #145929;
      border-bottom: 2px solid #42d56e;
    }
    .head2__note {
      padding: 5px 10px 0px;
      font-size: 14px;
      font-style: italic;
      color: #485536;
    }

    .head3 {
      margin-top: 5px;
      padding: 3px 40px 3px;
      border-left: 2px dotted #485536;
    }
    .head3__name {
      display: inline-block;
      max-width: 70%;
      padding: 2px 10px;
      font-size: 20px;
      font-weight: bold;
      color: #145929;
      border-bottom: 2px dotted #485536;
    }
    .head3__note {
      padding: 2px 10px 0px;
      font-size: 12px;
      font-style: italic;
      color: #485536;
    }

    .warn {
      background-color: #fcf2f2;
      border-color: #dfb5b4;
      border-left: 5px solid #dfb5b4;
      padding: 0.5em;
    }
    .note {
      color: #42d56e;
    }
    .note::before {
      content: 'Note. ';
      font-weight: bold;
      color: #485536;
    }

    .end {
      padding: 5px;
      border-top: 5px solid #8b1d1d;
      border-bottom: 5px solid #8b1d1d;
    }
'''

def init_jupyter():
    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
    print('Start | %s |'%t)
    print('-'*37)

    from IPython.core.display import HTML
    return HTML('<style>%s</style>' %css)

def func_check(N_tt, N_np, GR, func):
    from . import Func

    d = GR.d
    T_tt, E_tt, C_f_tt, C_a_tt = [], [], [], []
    T_np, E_np = [], []

    if N_tt is not None:
        for n in (N_tt):
            GR.n = np.array([n]*d)
            IT = Func(GR, eps=1.E-6, with_tt=True, log_path='./tmp.txt')
            IT.init(func).prep().calc().test()
            T_tt.append(IT.tms.copy())
            E_tt.append(np.mean(IT.err))
            c_f = IT.res['evals']
            for n in list(GR.n): c_f /= n
            C_f_tt.append(c_f)
            c_a = sum([G.size for G in tt.tensor.to_list(IT.A)])
            for n in list(GR.n): c_a /= n
            C_a_tt.append(c_a)

    if N_np is not None:
        for n in N_np:
            GR.n = np.array([n]*d)
            IT = Func(GR, eps=1.E-6, with_tt=False, log_path='./tmp.txt')
            IT.init(func).prep().calc().test()
            T_np.append(IT.tms.copy())
            E_np.append(np.mean(IT.err))

    fig = plt.figure(figsize=(12, 6))
    gs = mpl.gridspec.GridSpec(
        ncols=3, nrows=1, left=0.01, right=0.99, top=0.99, bottom=0.01,
        wspace=0.3, hspace=0.01, width_ratios=[1, 1, 1], height_ratios=[1]
    )

    ax = fig.add_subplot(gs[0, 0])
    if N_tt is not None:
        ax.plot(
            N_tt, [T['prep'] for T in T_tt], label='Prep (TT)',
            color='blue', linewidth=3
        )
        ax.plot(
            N_tt, [T['calc'] for T in T_tt], label='Calc (TT)',
            color='green', linewidth=3
        )
        ax.plot(
            N_tt, [T['comp'] for T in T_tt], label='Comp (TT)',
            color='orange', linewidth=3
        )
    if N_np is not None:
        ax.plot(
            N_np, [T['prep'] for T in T_np], '-.', label='Prep (NP)',
            color='blue', linewidth=1, marker='*', markersize=5
        )
        ax.plot(
            N_np, [T['calc'] for T in T_np], '-.', label='Calc (NP)',
            color='green', linewidth=1, marker='*', markersize=5
        )
        ax.plot(
            N_np, [T['comp'] for T in T_np], '-.', label='Comp (NP)',
            color='orange', linewidth=1, marker='*', markersize=5
        )
    ax.set_title('Times (%d-dim)'%d)
    ax.semilogx()
    ax.semilogy()
    if N_tt is not None and N_np is not None: ax.legend(loc='best')

    ax = fig.add_subplot(gs[0, 1])
    if N_tt is not None:
        ax.plot(
            N_tt, E_tt, label='TT',
            color='black', linewidth=3
        )
    if N_np is not None:
        ax.plot(
            N_np, E_np, '-.', label='NP',
            color='black', linewidth=1, marker='*', markersize=5
        )
    ax.set_title('Interpolation error (%d-dim)'%d)
    ax.semilogx()
    ax.semilogy()
    if N_tt is not None and N_np is not None: ax.legend(loc='best')

    ax = fig.add_subplot(gs[0, 2])
    if N_tt is not None:
        ax.plot(
            N_tt, C_f_tt, label='Function evals',
            color='brown', linewidth=3
        )
        ax.plot(
            N_tt, C_a_tt, label='Tensor size',
            color='magenta', linewidth=3
        )
        ax.set_title('Compression (tt) factor (%d-dim)'%d)
        ax.semilogx()
        ax.semilogy()
        ax.legend(loc='best')

    plt.show()
