"""Sympy numerical code generation utilities."""


import collections
import functools
import inspect
import itertools
import re

import numpy as np
import sympy


try:
    from cached_property import cached_property
except ModuleNotFoundError:
    def cached_property(f):
        """On-demand property which is calculated only once and memorized."""
        return property(functools.lru_cache()(f))


class cached_class_property:
    """Decorator to cache class properties."""
    def __init__(self, getter):
        functools.update_wrapper(getter, self)
        self.getter = getter
    
    def __get__(self, obj, cls=None):
        if hasattr(self, 'value'):
            return self.value
        if cls is None:
            cls = type(obj)
        self.value = self.getter(cls)
        return self.value


def init_static_variable(f):
    return f()


def ew_diff(ndexpr, *wrt, **kwargs):
    """Element-wise symbolic derivative of n-dimensional array-like expression.
    
    >>> import sympy
    >>> x = sympy.symbols('x')
    >>> ew_diff([[x**2, sympy.cos(x)], [5/x + 3, x**3 +2*x]], x)
    array([[2*x, -sin(x)],
           [-5/x**2, 3*x**2 + 2]], dtype=object)
    
    """
    out = np.empty_like(ndexpr, object)
    for ind, expr in np.ndenumerate(ndexpr):
        out[ind] = sympy.diff(expr, *wrt, **kwargs)
    return out


def ndexpr_diff(ndexpr, wrt):
    """Calculates the derivatives of an array expression w.r.t. to an ndarray.
    
    >>> from sympy import var, sin; from numpy import array
    >>> tup = var('x,y,z')
    >>> ndexpr_diff(tup, [x,y])
    array([[1, 0, 0],
           [0, 1, 0]], dtype=object)
    
    >>> ndexpr_diff([x**2+2*y/z, sin(x)], (x,y))
    array([[2*x, cos(x)],
           [2/z, 0]], dtype=object)
    
    """
    ndexpr = np.asarray(ndexpr)
    wrt = np.asarray(wrt)
    jac = np.empty(wrt.shape + ndexpr.shape, dtype=object)
    for i, elem in np.ndenumerate(wrt):
        diff = ew_diff(ndexpr, elem)
        jac[i] = diff if diff.shape else diff[()]
    return jac


def flat_cat(*args, **kwargs):
    """Concatenate flattened arrays."""
    chain = list(itertools.chain(args, kwargs.values()))
    if not chain:
        return np.array([])
    else:
        return np.concatenate([np.asanyarray(a).flatten() for a in chain])


def make_signature(arg_name_list, member=False):
    """Make Signature object from argument name list or str."""
    parameters = []
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    
    if member:
        parameters.append(inspect.Parameter('self', kind))
    
    if isinstance(arg_name_list, str):
        arg_name_list = map(str.strip, arg_name_list.split(','))

    for arg_name in arg_name_list:
        parameters.append(inspect.Parameter(arg_name, kind))
    
    return inspect.Signature(parameters)


class SymbolicSubsFunction:
    def __init__(self, arguments, output):
        self.arguments = tuple(arguments)
        self.output = output

    def __call__(self, *args):
        assert len(args) == len(self.arguments)
        subs = {}
        for var, value in zip(self.arguments, args):
            subs.update(var.subs_dict(value))
        
        # double substitution is needed when the same symbol appears in the
        # function definition and call arguments
        temp_subs = {s: sympy.Symbol('_temp_subs_' + s.name) for s in subs}
        final_subs = {temp_subs[s]: subs[s] for s in temp_subs}
        return self.output.subs(temp_subs).subs(final_subs)
