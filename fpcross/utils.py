import time
import platform
from tqdm import tqdm

from . import config

def ij():
    t = time.strftime('%l:%M%p %Z on %b %d, %Y')
    print('Start | %s | python %-8s |'%(t, platform.python_version()))
    print('-'*55)

    from IPython.core.display import HTML
    return HTML('<style>%s</style>'%config['css'])

def tms(name):
    '''
    Decorator. Save time (duration) for function call inside the class.
    The corresponding class should have tms dict with field name.
    The filed tms[name] (if exists) will be incremented by duration.
    * Will return class instance, not result of the decorated function.
    '''

    def timer_(f):
        def timer__(self, *args, **kwargs):
            t = time.perf_counter()
            r = f(self, *args, **kwargs)
            t = time.perf_counter() - t
            if hasattr(self, 'tms') and name in self.tms: self.tms[name]+= t
            return self # r
        return timer__
    return timer_

class PrinterSl(object):
    '''
    Present (print in interactive mode) current calculation status
    for the solver (fpcross.Solver).
    '''

    def __init__(self, SL, with_print=False):
        self.SL, self.with_print, self.tqdm = SL, with_print, None

    def init(self):
        d, u, t = 'Solve', 'step', self.SL.TG.n0 - 1
        if self.with_print: self.tqdm = tqdm(desc=d, unit=u, total=t, ncols=80)
        return self

    def update(self, msg=None):
        if self.with_print and msg: self.tqdm.set_postfix_str(msg, refresh=True)
        if self.with_print: self.tqdm.update(1)
        return self

    def close(self):
        if self.with_print: self.tqdm.close()
        return self
