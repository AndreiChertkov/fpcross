import time
import platform
from tqdm import tqdm

from . import config


def ij():
    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
    v = platform.python_version()
    print('Start | %s | python %-8s |\n'%(t, v) + '-'*55)

    from IPython.core.display import HTML
    return HTML('<style>%s</style>'%config['css'])


def tms(name, with_list=False):
    '''
    @Decorator. Save time (duration) for function call inside the class.
    The corresponding class may have tms dict with the field name (tms[name]),
    which (if exists) will be incremented by duration.
    * Will return class instance (not result of the decorated function!!!)
    * for the functions with special names: "init", "prep" and "calc".

    INPUT:

    name - name of the operation
    type: str

    with_list - flag:
        True  - duration will be also saved to the list in tms_list if exists
        False - duration will not be saved to the list
    type: bool

    TODO Add check that tms and tms_list are dicts.
    TODO Add doc for tms_list.
    '''

    def timer_(f):
        def timer__(self, *args, **kwargs):
            t = time.perf_counter()
            r = f(self, *args, **kwargs)
            t = time.perf_counter() - t

            if True:
                if hasattr(self, 'tms') and name in self.tms:
                    self.tms[name]+= t

            if with_list:
                if hasattr(self, 'tms_list') and name in self.tms_list:
                    self.tms_list[name].append(t)

            return self if f.__name__ in ['init', 'prep', 'calc'] else r

        return timer__

    return timer_


class PrinterSl(object):
    '''
    Present (print in interactive mode) current calculation status
    for the solver (fpcross.Solver).

    TODO Check displayed iteration number n0-1.
    '''

    def __init__(self, SL, with_print=False):
        self.SL = SL
        self.with_print = with_print
        self.tqdm = None

    def init(self):
        if self.with_print:
            d, u, t = 'Solve', 'step', self.SL.TG.n0 - 1
            self.tqdm = tqdm(desc=d, unit=u, total=t, ncols=80)

        return self

    def refr(self, msg=None):
        if self.with_print:
            if msg:
                self.tqdm.set_postfix_str(msg, refresh=True)
            self.tqdm.update(1)

        return self

    def close(self):
        if self.with_print:
            self.tqdm.close()

        return self
