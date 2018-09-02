"""Sympy numerical code generation utilities."""


import collections
import functools
import inspect
import itertools
import keyword
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


class classproperty:
    """Same as property(), but passes type(obj) instead of obj to methods."""
    
    def __init__(self, fget, doc=None):
        assert callable(fget)
        self.fget = fget
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc
    
    def __get__(self, obj, cls=None):
        if cls is None:
            cls = type(obj)
        return self.fget(cls)


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


def make_signature(arg_names, member=False):
    """Make Signature object from argument name iterable or str."""
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    
    if isinstance(arg_names, str):
        arg_names = map(str.strip, arg_name_list.split(','))
    if member and arg_names[0] != 'self':
        arg_names = ['self'] + arg_names
    
    return inspect.Signature([inspect.Parameter(n, kind) for n in arg_names])


def wrap_with_signature(arg_name_list, member=False):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args):
            return f(*args)
        wrapper.__signature__ = make_signature(arg_name_list, member)
        return wrapper
    return decorator


def sparsify(array, selector=None):
    """Get nonzero values and indices from an array."""
    rank = array.rank()

    values = []
    indices = []
    
    for index in np.ndindex(*array.shape):
        if selector is not None and not selector(*index):
            continue
        
        elem = array[index]
        if elem:
            values.append(elem)
            indices.append(index)
    
    if indices:
        return sympy.Array(values), np.transpose(indices)
    else:
        return sympy.Array([], 0), np.zeros((array.rank(), 0), int)


def istril(*index):
    """Return whether and index is in the lower triangle of an array."""
    return index[0] <= index[1]


def isstr(obj):
    """Return whether an object is instance of `str`."""
    return isinstance(obj, str)


def isiterable(obj):
    """Return whether an object is iterable."""
    return isinstance(obj, collections.Iterable)


def isidentifier(ident: str) -> bool:
    """Return whether a string is a valid python identifier."""
    
    if not isstr(ident):
        raise TypeError("expected str, but got {!r}".format(type(ident)))
    if not ident.isidentifier():
        return False
    if keyword.iskeyword(ident):
        return False
    
    return True
