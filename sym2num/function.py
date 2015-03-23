'''Sympy numeric function generation.'''


import collections
import itertools
import keyword
import re
import warnings

import numpy as np
import pystache
import sympy


function_template = '''\
def {{name}}(*_args, **_kwargs):
    _args = [{{numpy}}.asarray(arg) for arg in _args]
    _kwargs = {key: {{numpy}}.asarray(arg) for key, arg in _kwargs.items()}
    
    {{#args}}
    {{#elements}}
    {{elem_name}} = _args[{{arg_index}}][{{elem_index}}]
    {{/elements}}
    {{/args}}
    {{#kwargs}}
    {{#elements}}
    {{elem_name}} = _kwargs["{{key}}"][{{elem_index}}]
    {{/elements}}
    {{/kwargs}}

    _broadcast = {{numpy}}.broadcast(*{{broadcast_elems}})
    _base_shape = _broadcast.shape
    _out = {{numpy}}.empty(_base_shape + {{out_shape}})

    {{#out_elems}}
    _out[{{index}}] = {{expression}}
    {{/out_elems}}
    
    return _out
'''


def symbol_array(obj):
    # Convert to python object array
    arr = np.array(obj, dtype=object)
    
    # Check array contents
    for index, elem in np.ndenumerate(arr):
        if not isinstance(elem, sympy.Symbol):
            msg = "Array elements should be `sympy.Symbol`, got `{}.{}`."
            msg = msg.format(type(elem).__module__, type(elem).__name__)
            raise TypeError(msg)
        if keyword.iskeyword(elem.name):
                msg = "Element index `{}` is python keyword `{}`."
                raise ValueError(msg.format(index, elem.name))
        if not elem.name.isidentifier():
            msg = "Element `{}` at index `{}` is not a valid python identifier."
            raise ValueError(msg.format(index, elem.name))
        if elem.name.startswith('_'):
            warnings.warn("Symbols starting with '_' may conflict " +
                          "with internal generated variables.")
    
    return arr


class SymbolicFunction:
    def __init__(self, name, out, args=[], kwargs={}):
        self.args = [symbol_array(arg) for arg in args]
        self.kwargs = {key: symbol_array(arg) for key, arg in kwargs.items()}
        self.out = np.asarray(out, dtype=object)
        self.name = name
    
    def argument_tags(self, arg):
        return [
            dict(elem_index=((...,) + index), elem_name=elem.name)
            for (index, elem) in np.ndenumerate(arg)
        ]
    
    def print_def(self, printer):
        # Check for conflicts between function and printer symbols
        for arg in itertools.chain(self.args, self.kwargs.values()):
            for elem in arg.flat:
                if elem.name in printer.modules:
                    msg = "Function argument {} conflicts with printer module."
                    raise ValueError(msg.format(elem.name))
        
        # Create the template substitution tags
        tags = dict(name=self.name, numpy=printer.numpy, 
                    out_shape=self.out.shape)
        tags['args'] = [dict(arg_index=index, elements=self.argument_tags(arg))
                        for index, arg in enumerate(self.args)]
        tags['kwargs'] = [dict(key=key, elements=self.argument_tags(arg))
                          for key, arg in self.kwargs.items()]
        tags['broadcast_elems'] = tuple(
            a.flat[0] for a in itertools.chain(self.args, self.kwargs.values())
            if a.size > 0
        )
        tags['out_elems'] = [
            dict(index=((...,) + index), expression=printer.doprint(expr))
            for index, expr in np.ndenumerate(self.out)
        ]
        
        return pystache.render(function_template, tags)

