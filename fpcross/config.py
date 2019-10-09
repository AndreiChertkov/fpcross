# import platform
# print ( '  Python version: %s' % ( platform.python_version ( ) ) )

config = {
    'plot': {
        'time': {
            'fig': 'base_1_2',
            'grid': 'base_1_2',
            'title-sol': 'Solution',
            'title-err': 'Error',
            'label-sol': ['Time', 'Solution'],
            'label-err': ['Time', 'Error'],
            'line-sol-init': ['l1', 'Initial'],
            'line-sol-calc': ['l2', 'Calculated'],
            'line-sol-real': ['l3', 'Analytic'],
            'line-sol-stat': ['l4', 'Stationary'],
            'line-err-real': ['l5', 'Error vs analytic'],
            'line-err-stat': ['l6', 'Error vs stationary'],
        },
        'spatial': {
            'fig': 'base_1_2',
            'grid': 'base_1_2',
            'title-sol': 'Solution',
            'title-err': 'Error',
            'label-sol': ['Spatial point', 'Solution'],
            'label-err': ['Spatial point', 'Error'],
            'line-sol-init': ['l1', 'Initial'],
            'line-sol-calc': ['l2', 'Calculated'],
            'line-sol-real': ['l3', 'Analytic'],
            'line-sol-stat': ['l4', 'Stationary'],
            'line-err-real': ['l5', 'Error vs analytic'],
            'line-err-stat': ['l6', 'Error vs stationary'],
        },
        'conv': {
            'fig': 'base_1_2',
            'grid': 'base_1_2',
            'title-err': 'Error',
            'title-time': 'Computation time',
            'line-real': ['l2', ''],
            'line-appr': ['l4', ''],
        },
        'conv-all': {
            'fig': 'base_1_2',
            'grid': 'base_1_2',
            'title-err': 'Error',
            'title-time': 'Computation time',
            'line': ['l2', ''],
            'cols': ['#48484c', '#8b1d1d', '#5f91ac'],
        },
    },
    'opts': {
        'plot': {
            'fig': {
                'base_1_2': {
                    'figsize': (10, 5),
                },
                'base_1_3': {
                    'figsize': (12, 5),
                },
            },
            'grid': {
                'base_1_2': {
                    'ncols': 2,
                    'nrows': 1,
                    'left': 0.01,
                    'right': 0.99,
                    'top': 0.99,
                    'bottom': 0.01,
                    'wspace': 0.2,
                    'hspace': 0.1,
                    'width_ratios': [1, 1],
                    'height_ratios': [1],
                },
                'base_1_3': {
                    'ncols': 3,
                    'nrows': 1,
                    'left': 0.,
                    'right': 1.,
                    'top': 1.,
                    'bottom': 0.,
                    'wspace': 0.2,
                    'hspace': 0.,
                    'width_ratios': [1, 1, 1],
                    'height_ratios': [1],
                },
            },
            'line': {
                'l1': {
                    'linestyle': '--',
                    'linewidth': 3,
                    'color': '#48484c',
                    'marker': 'o',
                    'markersize': 0,
                    'markerfacecolor': '#48484c',
                    'markeredgewidth': 0,
                    'markeredgecolor': '#48484c',
                },
                'l2': {
                    'linestyle': '-',
                    'linewidth': 1,
                    'color': '#8b1d1d',
                    'marker': 'o',
                    'markersize': 7,
                    'markerfacecolor': '#8b1d1d',
                    'markeredgewidth': 1,
                    'markeredgecolor': '#8b1d1d',
                },
                'l3': {
                    'linestyle': '-',
                    'linewidth': 3,
                    'color': '#485536',
                    'marker': 'o',
                    'markersize': 0,
                    'markerfacecolor': '#485536',
                    'markeredgewidth': 0,
                    'markeredgecolor': '#485536',
                },
                'l4': {
                    'linestyle': '-.',
                    'linewidth': 2,
                    'color': '#5f91ac',
                    'marker': 'o',
                    'markersize': 0,
                    'markerfacecolor': '#5f91ac',
                    'markeredgewidth': 0,
                    'markeredgecolor': '#5f91ac',
                },
                'l5': {
                    'linestyle': '-',
                    'linewidth': 2,
                    'color': '#485536',
                    'marker': 'o',
                    'markersize': 6,
                    'markerfacecolor': '#8b1d1d',
                    'markeredgewidth': 1,
                    'markeredgecolor': '#8b1d1d',
                },
                'l6': {
                    'linestyle': '-.',
                    'linewidth': 2,
                    'color': '#5f91ac',
                    'marker': '*',
                    'markersize': 6,
                    'markerfacecolor': '#5f91ac',
                    'markeredgewidth': 1,
                    'markeredgecolor': '#5f91ac',
                },
            },
        },
    },
    'css': '''
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
    ''',
}
