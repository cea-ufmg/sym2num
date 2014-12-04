'''Sympy printer for numeric code generation.'''


import itertools
import keyword
import re
import warnings

import numpy as np
import pystache
import sympy
from sympy import printing


class NumpyPrinter(printing.str.StrPrinter):
    '''Sympy printer for generating python code using numpy.'''
    
    printmethod = 'numpyrepr'
    
    def __init__(self, numpy="numpy"):
        super().__init__()        
        self.numpy = numpy
        '''The module name of numpy.'''

    @property
    def modules(self):
        return (self.numpy,)
    
    def _print_atan2(self, expr):
        return '%s.arctan2(%s)' % (self.numpy, self.stringify(expr.args, ', '))
    
    def _print_Function(self, expr):
        return '%s.%s' % (self.numpy, super()._print_Function(expr))
    
    def _print_Pi(self, expr):
        return '%s.pi' % (self.numpy,)
    
    def _print_Pow(self, expr):
        if isinstance(expr.args[1], numbers.Half):
            return '%s.sqrt(%s)' % (self.numpy, self._print(expr.args[0]))
        else:
            return super()._print_Pow(expr)


class ScipyPrinter(NumpyPrinter):
    '''Sympy printer for generating python code using scipy.'''
    
    def __init__(self, scipy="scipy", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scipy = scipy
        '''The module name of scipy.'''

    @property
    def modules(self):
        return super().modules + (self.scipy,)
    
    def _print_erf(self, expr):
        return '%s.special.erf(%s)' % (self.scipy, self._print(expr.args[0]))
    
    def _print_loggamma(self, expr):
        return '%s.special.gammaln(%s)' % (self.scipy,self._print(expr.args[0]))



function_template = '''\
def {{name}}(*args, **kwargs):
    {{#args}}
    {{contents}} = {{numpy}}.asarray(args[{{index}}])
    {{/args}}

    {{#kwargs}}
    {{contents}} = {{numpy}}.asarray(kwargs["{{key}}"])
    {{/kwargs}}

    _bcast_args = {{numpy}}.broadcast_arrays(*{{broadcast_elems}})
    _base_shape = _bcast_args[0].shape
    _out = {{numpy}}.empty({{out_shape}} + _base_shape)

    {{#out_elems}}
    _out[{{index}}] = {{expression}}
    {{/out_elems}}
    
    return _out
'''


def function_def(printer, name, out, args=[], kwargs={}):
    # Ensure inputs are ndarrays
    args = [np.asarray(arg, dtype=object) for arg in args]
    kwargs = {key: np.asarray(arg, dtype=object) for (key, arg) in kwargs}
    out = np.asarray(out, dtype=object)
    
    # Ensure arguments are valid
    for arg in itertools.chain(args, kwargs.values()):
        for elem in arg.flat:
            if not isinstance(elem, sympy.Symbol):
                raise TypeError(
                    "Function arguments should be sympy.Symbol arrays.", elem
                )
            if keyword.iskeyword(elem.name):
                raise ValueError(
                    "Function arguments cannot be python keywords.", elem
                )
            if not elem.name.isidentifier():
                raise ValueError(
                    "Function argument is not a valid python identifier.", elem
                )
            if elem.name in printer.modules:
                raise ValueError(
                    "Function argument conflicts with printer module.", elem
                )
            if elem.name.startswith('_'):
                warnings.warn("Symbols starting with '_' may conflict with " +
                              "internal function variables.")
    
    # Create the template substitution tags
    tags = dict(name=name, numpy=printer.numpy, out_shape=str(out.shape))
    tags['args'] = [dict(index=index, contents=arg.tolist())
                    for index, arg in enumerate(args)]
    tags['kwargs'] = [dict(key=key, contents=arg.tolist())
                      for key, arg in kwargs.items()]
    tags['broadcast_elems'] = tuple(
        arg.flat[0] for arg in itertools.chain(args, kwargs.values())
    )
    tags['out_elems'] = [dict(index=index, expression=printer.doprint(expr))
                         for index, expr in np.ndenumerate(out)]
    
    return pystache.render(function_template, tags)


def indent(text, prefix=(4 * ' ')):
    '''Indent a multiline string.'''
    return re.sub('^', prefix, text, flags=re.MULTILINE)
