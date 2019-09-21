from importlib import import_module

from model_ import Model as ModelBase

class Model(object):

    def __init__(self, name=None):
        self.mds = None
        self.mds = None

        if name is not None: self.select(name)

    def select(self, name):
        self.md = import_module('model_%s'%name).Model()

    def init(self, *args, **kwargs):
        '''
        Set and prepare (if is needed) parameters of the model.
        '''

        self._check_is_selected()
        return self.md.init(*args, **kwargs)

    def info(self):
        '''
        Present info about the model.
        Latex mode for jupyter lab cells is used.
        '''

        self._check_is_selected()
        return self.md.info()

    def f0(self, x, t):
        '''
        Function f(x, t).

        INPUT:

        x - values of the spatial variable
        type: ndarray (or list) [dimensions, number of points] of float

        t - value of the time variable
        type: float

        OUTPUT:

        v - corresponding values
        type: ndarray [dimensions, number of points] of float
        '''

        self._check_is_selected()
        return self.md.f0(x, t)

    def f1(self, x, t):
        '''
        Spatial derivative d f(x, t) / d x.

        INPUT:

        x - values of the spatial variable
        type: ndarray (or list) [dimensions, number of points] of float

        t - value of the time variable
        type: float

        OUTPUT:

        v - corresponding values
        type: ndarray [dimensions, number of points] of float
        '''

        self._check_is_selected()
        return self.md.f1(x, t)

    def r0(self, x):
        '''
        Initial condition r0(x).

        INPUT:

        x - values of the spatial variable
        type: ndarray (or list) [dimensions, number of points] of float

        OUTPUT:

        v - corresponding values
        type: ndarray [number of points] of float
        '''

        self._check_is_selected()
        return self.md.r0(x)

    def rt(self, x, t):
        '''
        Real (analytic) solution r(x, t) if known.

        INPUT:

        x - values of the spatial variable
        type: ndarray (or list) [dimensions, number of points] of float

        t - value of the time variable
        type: float

        OUTPUT:

        v - corresponding values
        type: ndarray [number of points] of float
        '''

        self._check_is_selected()
        return self.md.rt(x, t)

    def rs(self, x):
        '''
        Stationary (analytic) solution rs(x) if known.

        INPUT:

        x - values of the spatial variable
        type: ndarray (or list) [dimensions, number of points] of float

        OUTPUT:

        v - corresponding values
        type: ndarray [number of points] of float
        '''

        self._check_is_selected()
        return self.md.rs(x)

    def _check_is_selected(self):
        if self.md is not None: return True

        s = 'The model is not selected.'
        raise ValueError(s)

# model_fpe_1d_oup

#p_fold = os.path.dirname(os.path.abspath(__file__))
#p_file = os.path.join(p_fold, './model%d.md'%self.numb)
#with open(p_file) as f: info = f.read()
