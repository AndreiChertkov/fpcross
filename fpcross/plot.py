import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


COLOR1 = '#8b1d1d'
COLOR2 = '#5f91ac'


def plot(res, fpath=None):
    t = [i*res.eq.h for i in range(1, res.eq.m + 1)]

    if res.is_full:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.2)

    ax1.set_title('Relative error')
    ax1.set_xlabel('Time')
    if len(res.es_list) == len(t):
        ax1.plot(t, res.es_list, label='vs stationary',
            linestyle='-', linewidth=2, color=COLOR1,
            marker='o', markersize=7, markerfacecolor=COLOR1,
            markeredgewidth=1, markeredgecolor=COLOR1)
    if len(res.et_list) == len(t):
        ax1.plot(t, res.et_list, label='vs analytic',
            linestyle='-.', linewidth=2, color=COLOR2,
            marker='o', markersize=0, markerfacecolor=COLOR2,
            markeredgewidth=0, markeredgecolor=COLOR2)
    prep(ax1)

    if not res.is_full:
        ax2.set_title('TT-rank')
        ax2.set_xlabel('Time')
        ax2.plot(t, res.r_list,
            linestyle='-', linewidth=2, color=COLOR1,
            marker='o', markersize=7, markerfacecolor=COLOR1,
            markeredgewidth=1, markeredgecolor=COLOR1)
        prep(ax2, with_leg=False, is_log=False)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_spec(res, fpath=None):
    t = [i*res.eq.h for i in range(1, res.eq.m + 1)]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.2)

    with_psi = len(res.eq.psi_list) == len(t)
    with_eta = len(res.eq.eta_list) == len(t)

    ax1.set_title('Computation results')
    ax1.set_xlabel('Time')
    if with_psi:
        ax1.plot(t, res.eq.psi_list, label='Value of $\psi$',
            linestyle='-', linewidth=2, color='#5f91ac',
            marker='o', markersize=7, markerfacecolor='#5f91ac',
            markeredgewidth=1, markeredgecolor='#5f91ac')
    if with_eta:
        ax1.plot(t, res.eq.eta_list, label='Value of $\eta$',
            linestyle='-', linewidth=2, color='#8b1d1d',
            marker='o', markersize=7, markerfacecolor='#8b1d1d',
            markeredgewidth=1, markeredgecolor='#8b1d1d')
    prep(ax1, with_leg=(with_psi or with_eta), is_log=False)

    ax2.set_title('TT-rank')
    ax2.set_xlabel('Time')
    ax2.plot(t, res.r_list,
        linestyle='-', linewidth=2, color='#8b1d1d',
        marker='o', markersize=7, markerfacecolor='#8b1d1d',
        markeredgewidth=1, markeredgecolor='#8b1d1d')
    prep(ax2, with_leg=False, is_log=False)

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def prep(ax, with_leg=True, is_log=True, is_int_x=False):
    if with_leg:
        ax.legend(loc='best', frameon=True)
    if is_log:
        ax.semilogy()
    if is_int_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(ls=":")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
