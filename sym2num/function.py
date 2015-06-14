'''Sympy numeric function generation.'''


import collections
import collections.abc as colabc
import itertools
import keyword
import re
import warnings

import numpy as np
import pystache
import sympy

from . import utils


function_template = '''\
def {{name}}({{signature}}):
    """Generated function `{{name}}` from sympy ndarray expression."""
    # Convert arguments to ndarrays and create aliases to prevent name conflicts
    {{#args}}
    _arg_{{arg_name}} = {{numpy}}.asarray({{arg_name}})
    {{/args}}
    
    # Check argument lengths
    {{#args}}
    {{#arg_shape}}
    if _arg_{{arg_name}}.shape[-{{ndim}}:] != {{arg_shape}}:
        raise ValueError("Invalid dimensions for argument `{{arg_name}}`.")
    {{/arg_shape}}
    {{/args}}
    
    # Unpack the elements of each argument
    {{#args}}
    {{#elements}}
    {{elem_name}} = _arg_{{arg_name}}[{{elem_index}}]
    {{/elements}}

    {{/args}}
    # Broadcast the input arguments
    {{^single_arg}}
    _broadcast = {{numpy}}.broadcast({{broadcast}})
    _base_shape = _broadcast.shape
    {{/single_arg}}
    {{#single_arg}}
    _base_shape = {{broadcast}}.shape
    {{/single_arg}}
    _out = {{numpy}}.zeros(_base_shape + {{out_shape}})

    # Assign the nonzero elements of the output
    {{#out_elems}}
    _out[{{index}}] = {{expr}}
    {{/out_elems}}
    
    return _out'''


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
    def __init__(self, f, args={}, name=None):
        # Process and save the input arguments
        arg_items = args.items() if isinstance(args, colabc.Mapping) else args
        arg_items = [(name, symbol_array(arg)) for name, arg in arg_items]
        self.args = collections.OrderedDict(arg_items)
        self.out = np.asarray(f(*self.args.values()) if callable(f) else f)
        self.name = name or (f.__name__ if callable(f) else None)
        
        # Check if a valid name was found
        if name is None and not callable(f):
            raise TypeError("A name must be provided if f is not callable.")
        if not self.name:
            raise ValueError("Invalid function name for symbolic function.")
        
        # Check for duplicate symbols
        arg_elements = utils.flat_cat(**self.args)
        unique_arg_elements = set(arg_elements)
        if len(unique_arg_elements) != arg_elements.size:
            raise ValueError('Duplicate symbols found in function arguments.')
    
    def argument_tags(self, arg):
        return [
            dict(elem_index=((...,) + index), elem_name=elem.name)
            for (index, elem) in np.ndenumerate(arg)
        ]
    
    def print_def(self, printer):
        # Check for conflicts between function and printer symbols
        for elem in utils.flat_cat(**self.args):
            if elem.name in printer.modules:
                msg = "Function argument {} conflicts with printer module."
                raise ValueError(msg.format(elem.name))
        
        # Create the template substitution tags
        broadcast = comma_join(
            a.flat[0] for a in self.args.values() if a.size > 0
        )
        
        tags = {
            'name': self.name,
            'numpy': printer.numpy,
            'broadcast': broadcast,
            'single_arg': len(self.args) == 1,
            'out_shape': self.out.shape,
            'signature': comma_join(self.args.keys()),
            'args': [],
            'out_elems': [dict(index=((...,) + i), expr=printer.doprint(expr))
                          for i, expr in np.ndenumerate(self.out) if expr != 0]
        }
        for name, arg in self.args.items():
            arg_tags = dict(arg_name=name, ndim=arg.ndim, arg_shape=arg.shape,
                            elements=self.argument_tags(arg))
            tags['args'].append(arg_tags)
        
        # Render and return
        return pystache.render(function_template, tags)
    
    def diff(self, wrt, name):
        diff = utils.ndexpr_diff(self.out, wrt)
        return type(self)(diff, args=self.args, name=name)


def comma_join(iterable):
    return ", ".join(str(x) for x in iterable)

