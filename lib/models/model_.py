import os
import numpy as np

class Model(object):
    '''
    Base (abstract) class for all models of equations.
    '''

    def __init__(self, name, desc, tags, info):
        '''
        Set description of the model.
        '''

        self._name = name
        self._desc = desc
        self._tags = tags
        self._info = info

        self.d = 0

        self.init()

    def init(self):
        '''
        Set and prepare (if is needed) parameters of the model.
        '''

        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)

    def info(self):
        '''
        Present info about the model.
        Latex mode for jupyter lab cells is used.
        '''

        from IPython.display import display, Latex
        display(Latex(self._info['latex']))

    def dim(self):
        '''
        Dimension of the equation.

        OUTPUT:

        v - corresponding value
        type: int, >= 1
        '''

        return self.d

    def d0(self):
        '''
        Diffusion coefficient D.

        OUTPUT:

        v - corresponding value
        type: float
        '''

        return self._d0()

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

        x = self._prep_x(x)
        return self._f0(x, t)

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

        x = self._prep_x(x)
        return self._f1(x, t)

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

        x = self._prep_x(x)
        return self._r0(x)

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

        x = self._prep_x(x)
        return self._rt(x, t)

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

        x = self._prep_x(x)
        return self._rs(x)

    def with_rt(self):
        '''
        Check if model has real (analytic) solution r(x, t).

        OUTPUT:

        v - corresponding flag
        type: bool
        '''

        return self._with_rt()

    def with_rs(self):
        '''
        Check if model has stationary (analytic) solution rs(x).

        OUTPUT:

        v - corresponding flag
        type: bool
        '''

        return self._with_rs()

    def _set(self, name, v, v0):
        if v is None: v = v0
        object.__setattr__(self, name, v)

    def _prep_x(self, x):
        if isinstance(x, list): x = np.array(x)

        return x

    def _d0(self):
        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)

    def _f0(self, x, t):
        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)

    def _f1(self, x, t):
        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)

    def _r0(self, x):
        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)

    def _rt(self, x, t):
        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)

    def _rs(self, x):
        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)

    def _with_rt(self):
        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)

    def _with_rs(self):
        s = 'Is abstract model. Use method of the specific model.'
        raise NotImplementedError(s)
