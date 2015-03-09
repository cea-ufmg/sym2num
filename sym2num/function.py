'''Sympy numeric function generation.'''


import itertools
import keyword
import re
import warnings

import numpy as np
import pystache
import sympy


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


class SymbolicFunction:
    def __init__(self, name, out, args=[], kwargs={}):
        # Ensure inputs are ndarrays
        args = [np.asarray(arg, dtype=object) for arg in args]
        kwargs = {key: np.asarray(arg, dtype=object) for (key, arg) in kwargs}
        out = np.asarray(out, dtype=object)
        
        # Ensure arguments are valid
        for arg in itertools.chain(args, kwargs.values()):
            for elem in arg.flat:
                if not isinstance(elem, sympy.Symbol):
                    msg = "Function arguments should be sympy.Symbol arrays."
                    raise TypeError(msg)
                if keyword.iskeyword(elem.name):
                    msg = "Function arguments cannot be python keywords."
                    raise ValueError(msg)
                if not elem.name.isidentifier():
                    msg = "Function argument is not a valid python identifier."
                    raise ValueError(msg)
                if elem.name in printer.modules:
                    msg = "Function argument conflicts with printer module."
                    raise ValueError(msg)
                if elem.name.startswith('_'):
                    warnings.warn("Symbols starting with '_' may conflict " +
                                  "with internal function variables.")
        self.args = args
        self.kwargs = kwargs
        self.out = out
    
    def print_def(self, printer):
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
