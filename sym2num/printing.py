'''Sympy printer for numeric code generation.'''


import re

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
    
    def _print_erf(self, expr):
        return '%s.special.erf(%s)' % (self.scipy, self._print(expr.args[0]))
    
    def _print_loggamma(self, expr):
        return '%s.special.gammaln(%s)' % (self.scipy,self._print(expr.args[0]))



function_template = '''\
def {{name}}(*args, **kwargs):
    {{#args}}
    {{contents}} = args[{{index}}]
    {{/args}}

    {{#kwargs}}
    {{contents}} = kwargs[{{key}}]
    {{/kwargs}}

    return
'''


def function_def(printer, name, out, args=[], kwargs={}):
    args = [np.asarray(arg, dtype=object) for arg in args]
    kwargs = {key: np.asarray(arg, dtype=object) for (key, arg) in kwargs}
    out = np.asarray(out, dtype=object)
    
    tags = dict(name=name, args=[], kwargs=[])
    
    for index, arg in enumerate(args):
        tags['args'].append(dict(index=index, contents=str(arg.tolist())))
    
    for key, arg in enumerate(kwargs.items()):
        tags['args'].append(dict(key=key, contents=printer.doprint(arg)))
    
    
    return pystache.render(function_template, tags)


def indent(text, prefix=(4 * ' ')):
    '''Indent a multiline string.'''
    return re.sub('^', prefix, text, flags=re.MULTILINE)
