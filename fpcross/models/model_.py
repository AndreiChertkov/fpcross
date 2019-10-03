import numpy as np

class Model(object):
    '''
    Base (abstract) class for all models of equations.
    '''

    def __init__(self, info):
        '''
        Set description of the model.
        '''

        self._info = info
        self.d = 0
        self.init()

    def init(self, *args, **kwargs):
        '''
        Set and prepare (if is needed) parameters of the model.
        '''

        for [name, opts] in self._info.get('pars', {}).items():
            object.__setattr__(self, name, kwargs.get(name, opts.get('dflt')))

    def info(self):
        '''
        Present info about the model.
        Markdown mode for jupyter lab cells is used.
        '''

        n = self._info['name']
        d = self._info['desc']
        t = self._info['tags']
        s = r'''<div class="head0">
            <div class="head0__name">Model problem</div>
            <div class="head0__note">%s :<br>%s [%s].</div>
        </div>'''%(n, d, ', '.join(t))

        ss = []
        for [name, opts] in self._info['pars'].items():
            ss.append('%s - %s (type: %s)'%(name, opts['name'], opts['type']))
        s+= r'''<div class="head2">
            <div class="head2__name">Parameters</div>
            <div class="head2__note"><ul><li>%s</li></ul></div>
        </div>'''%('</li><li>'.join(ss))

        s+= r'''<div class="head1">
            <div class="head1__name">Description</div>
        </div>'''
        s+= self._info['text']

        for note in self._info.get('notes', []):
            s+= r'''<div class="note">%s</div>'''%note

        s+= r'''<div class="end"></div>'''

        from IPython.display import display, Markdown
        display(Markdown(s))

    def dim(self):
        '''
        Dimension of the equation.

        OUTPUT:

        v - corresponding value
        type: int, >= 1
        '''

        return self.d

    def Dc(self):
        '''
        Diffusion coefficient D.

        OUTPUT:

        v - corresponding value
        type: float
        '''

        return self._Dc()

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

    def _prep_x(self, x):
        if isinstance(x, list): x = np.array(x)

        return x

    def _Dc(self):
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
