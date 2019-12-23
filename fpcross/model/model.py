import types
import numpy as np
from importlib import import_module


class Model(object):
    '''
    Base (abstract) class for all models of equations.
    Use static method Model.select to obtain predefined model by its name.
    '''

    def __init__(self):
        self.init()

    def init(self, **kwargs):
        '''
        Set model parameters.

        OUTPUT:

        MD - self
        type: fpcross.Model
        '''

        for [name, opts] in self.pars().items():
            if name in kwargs:
                v = kwargs[name]
            else:
                v = opts.get('dflt')
                if isinstance(v, types.FunctionType):
                    v = v(kwargs)

            setattr(self, '_' + name, v)

        return self

    def prep(self):
        '''
        Prepare special model parameters.

        OUTPUT:

        MD - self
        type: fpcross.Model
        '''

        return self

    def name(self):
        '''
        Model name (for compact presentation).

        OUTPUT:

        res - name
        type: str
        '''

        return 'model'

    def repr(self):
        '''
        Model equation in the string form (for compact presentation).

        OUTPUT:

        res - equation
        type: str
        '''

        return ''

    def desc(self):
        '''
        Model description in the string form (for compact presentation).

        OUTPUT:

        res - description
        type: str
        '''

        return ''

    def tags(self):
        '''
        Tags for the model.

        OUTPUT:

        res - tags
        type: list of str

        TODO Add the list of all supported tags.
        '''

        return []

    def pars(self):
        '''
        Description of the model parameters.

        OUTPUT:

        res - parameters
        type: dict
        * Each field is also a dict with fields 'name' (str), 'desc' (str),
        * 'type' (str), 'dflt' (type name or function), 'frmt' (str).
        '''

        return {}

    def coms(self):
        '''
        Comments for the model.

        OUTPUT:

        res - comments
        type: list of str (latex formulas may be used)
        '''

        return []

    def text(self):
        '''
        Model long representation by the text (markdown with latex format).

        OUTPUT:

        res - text (markdown with latex format)
        type: str
        '''

        return ''

    def info(self, is_comp=False, is_ret=False):
        '''
        Present info about the model.

        INPUT:

        is_comp - flag:
            True  - compact info will be presented (print method is used)
            False - full info will be presented (display function is used)
        type: bool

        is_ret - flag:
            True  - return string info
            False - print/display string info
        type: bool

        OUTPUT:

        s - (if is_ret) string with info
        type: str

        TODO Check replacement for indents.
        '''

        name = self.name() or '?????'
        repr = self.repr() or '?????'
        desc = self.desc() or '.....'
        tags = self.tags() or []
        pars = self.pars() or {}
        coms = self.coms() or []
        text = self.text() or ''

        if is_comp:
            s = 'Model : %-22s | %s\n'%(name, repr)
            s+= '>>>>>>> Description            : %s'%(desc or '...')

            if is_ret:
                return s + '\n'
            print(s)
            return

        s_name = r'<div class="head0__name">%s</div>'%name
        s_tags = r' [%s]'%(', '.join(tags)) if len(tags) else ''
        s_desc = r'<div class="head0__note">%s%s.</div>'%(desc, s_tags)
        s_main = r'<div class="head0">%s%s</div>'%(s_name, s_desc)

        s_pars = []
        for [name, opts] in pars.items():
            s_pars.append('%s = %s [%s]<div>%s (type: %s, default: %s)</div>'%(
                name, opts['frmt']%getattr(self, '_'+name), opts['name'],
                opts['desc'], opts['type'], opts['frmt']%opts['dflt']
            ))
        if len(s_pars):
            s_pars = r'''<div class="head2">
                <div class="head2__name">Parameters</div>
                <div class="head2__note"><ul><li>%s</li></ul></div>
            </div>'''%('</li><li>'.join(s_pars))
        else:
            s_pars = ''

        s_text = ''
        if text:
            s_text = r'''<div class="head1">
                <div class="head1__name">Description</div>
            </div>'''
            s_text+= text.replace('\n            ', '\n')

        s_coms = ''
        for com in coms:
            s_coms+= r'''<div class="note">%s</div>'''%com

        s = s_main + s_pars + s_text + s_coms + r'''<div class="end"></div>'''

        if is_ret:
            return s

        from IPython.display import display, Markdown
        display(Markdown(s))

    def d(self):
        '''
        Spatial dimension.
        '''

        raise NotImplementedError('This is abstract class.')

    def D(self):
        '''
        Diffusion coefficient.
        '''

        raise NotImplementedError('This is abstract class.')

    def f0(self, X, t):
        '''
        Function f(x, t).

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points]

        t - value of the time
        type: float

        OUTPUT:

        v - values
        type: ndarray [dimensions, number of points] of float
        '''

        raise NotImplementedError('This is abstract class.')

    def f1(self, X, t):
        '''
        Function d f_i(x, t) / d x_i (i = 1, 2, ..., d).

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points]

        t - value of the time
        type: float

        OUTPUT:

        v - values
        type: ndarray [dimensions, number of points] of float

        TODO Add method with_f1 for the case if derivative is not known.
        '''

        raise NotImplementedError('This is abstract class.')

    def r0(self, X):
        '''
        Initial condition.

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points]

        OUTPUT:

        v - values
        type: ndarray [number of points] of float
        '''

        raise NotImplementedError('This is abstract class.')

    def rt(self, X, t):
        '''
        Exact analytic solution.

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points]

        t - value of the time
        type: float

        OUTPUT:

        v - values
        type: ndarray [number of points] of float
        '''

        if not self.with_rt:
            raise ValueError('The model has not rt.')

        raise NotImplementedError('This is abstract class.')

    def rs(self, X):
        '''
        Exact stationary solution.

        INPUT:

        X - values of the spatial variable
        type: ndarray [dimensions, number of points]

        OUTPUT:

        v - values
        type: ndarray [number of points] of float
        '''

        if not self.with_rs:
            raise ValueError('The model has not rs.')

        raise NotImplementedError('This is abstract class.')

    def with_rt(self):
        '''
        Return True if model has known exact analytic solution.
        '''

        return False

    def with_rs(self):
        '''
        Return True if model has known exact stationary solution.
        '''

        return False

    @staticmethod
    def select(name):
        '''
        Select model by its name.

        INPUT:

        name - name of the model to select
        type: str
        * All '-' will be replaced by '_'.

        OUTPUT:

        MD - model
        type: fpcross.Model

        TODO Add try block.
        '''

        name = name.replace('-', '_')

        MD = import_module('.model.model_%s'%name, 'fpcross').Model()

        return MD
