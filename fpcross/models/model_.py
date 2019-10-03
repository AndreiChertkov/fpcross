import types
import numpy as np

class Model(object):
    '''
    Base (abstract) class for all models of equations.
    '''

    def __init__(self, info, d=0):
        '''
        Set description of the model.
        '''

        self._info = info
        self.d = d
        self.init()

    def init(self, *args, **kwargs):
        for [name, opts] in self._info.get('pars', {}).items():
            v = kwargs.get(name)
            if v is None:
                v = opts.get('dflt')
                if isinstance(v, types.FunctionType):
                    v = v(kwargs)
            object.__setattr__(self, name, v)

    def info(self):
        n = self._info['name']
        d = self._info['desc']
        t = self._info['tags']
        s = r'''<div class="head0">
            <div class="head0__name">%s</div>
            <div class="head0__note">%s [%s].</div>
        </div>'''%(n, d, ', '.join(t))

        ss = []
        for [name, opts] in self._info['pars'].items():
            ss.append('%s = %s [%s]<div>%s (type: %s, default: %s)</div>'%(
                name, opts['frmt']%getattr(self, name), opts['name'],
                opts['desc'], opts['type'], opts['frmt']%opts['dflt']
            ))
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
        return self.d

    def Dc(self):
        return self._Dc()

    def f0(self, x, t):
        return self._f0(self._prep_x(x), t)

    def f1(self, x, t):
        return self._f1(self._prep_x(x), t)

    def r0(self, x):
        return self._r0(self._prep_x(x))

    def rt(self, x, t):
        if not self.with_rt: raise ValueError('The model has not rt.')
        return self._rt(self._prep_x(x), t)

    def rs(self, x):
        if not self.with_rs: raise ValueError('The model has not rs.')
        return self._rs(self._prep_x(x))

    def with_rt(self):
        return getattr(self, '_rt', None) is not None

    def with_rs(self):
        return getattr(self, '_rs', None) is not None

    def _prep_x(self, x):
        if isinstance(x, list): x = np.array(x)
        return x
