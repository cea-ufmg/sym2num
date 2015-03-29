'''Sympy numeric function generation.'''


import collections
import itertools
import keyword
import re
import warnings

import numpy as np
import pystache
import sympy

from . import utils


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

    _broadcast = {{numpy}}.broadcast({{broadcast}})
    _base_shape = _broadcast.shape
    _out = {{numpy}}.zeros(_base_shape + {{out_shape}})

    {{#out_elems}}
    _out[{{index}}] = {{expr}}
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
    def __init__(self, f, args=[], kwargs={}, name=None):
        # Save the input arguments
        self.args = [symbol_array(arg) for arg in args]
        self.kwargs = {key: symbol_array(arg) for key, arg in kwargs.items()}
        self.out = f(*self.args, **self.kwargs) if callable(f) else f
        self.name = name or (f.__name__ if callable(f) else None)
        
        # Check for duplicate symbols
        arg_elements = utils.flat_cat(*self.args, **self.kwargs)
        unique_arg_elements = set(arg_elements)
        if len(unique_arg_elements) != arg_elements.size:
            raise ValueError('Duplicate symbols found in function arguments.')
    
    def argument_tags(self, arg):
        return [
            dict(elem_index=((...,) + index), elem_name=elem.name)
            for (index, elem) in np.ndenumerate(arg)
        ]
    
    def print_def(self, printer, name=None):
        # Initialize internal variables
        name = name or self.name or 'function'
        arg_chain = itertools.chain(self.args, self.kwargs.values())
        arg_elements = utils.flat_cat(*self.args, **self.kwargs)
        
        # Check for conflicts between function and printer symbols
        for elem in arg_elements:
            if elem.name in printer.modules:
                msg = "Function argument {} conflicts with printer module."
                raise ValueError(msg.format(elem.name))
        
        # Create the template substitution tags
        tags = {
            'name': name, 
            'numpy': printer.numpy,
            'out_shape': self.out.shape,
            'args': [dict(arg_index=index, elements=self.argument_tags(arg))
                     for index, arg in enumerate(self.args)],
            'kwargs': [dict(key=key, elements=self.argument_tags(arg))
                       for key, arg in self.kwargs.items()],
            'broadcast': comma_join(a.flat[0] for a in arg_chain if a.size > 0),
            'out_elems': [dict(index=((...,) + i), expr=printer.doprint(expr))
                          for i, expr in np.ndenumerate(self.out) if expr != 0]
        }
        
        # Render and return
        return pystache.render(function_template, tags)
    
    def diff(self, wrt, name=None):
        diff = utils.ndexpr_diff(self.out, wrt)
        return type(self)(diff, args=self.args, kwargs=self.kwargs, name=name)


def comma_join(iterable):
    return ", ".join(str(x) for x in iterable)

