'''Sympy numerical code generation utilities.'''


import itertools
import re

import numpy as np
import sympy


def ew_diff(ndexpr, *wrt, **kwargs):
    '''Element-wise symbolic derivative of n-dimensional array-like expression.
    
    >>> import sympy
    >>> x = sympy.symbols('x')
    >>> ew_diff([[x**2, sympy.cos(x)], [5/x + 3, x**3 +2*x]], x)
    array([[2*x, -sin(x)],
           [-5/x**2, 3*x**2 + 2]], dtype=object)
    
    '''
    out = np.empty_like(ndexpr, object)
    for ind, expr in np.ndenumerate(ndexpr):
        out[ind] = sympy.diff(expr, *wrt, **kwargs)
    
    return out


def ndexpr_diff(ndexpr, wrt):
    '''Calculates the derivatives of an array expression w.r.t. to an ndarray.
    
    >>> from sympy import var, sin; from numpy import array
    >>> tup = var('x,y,z')
    >>> ndexpr_diff(tup, [x,y])
    array([[1, 0],
           [0, 1],
           [0, 0]], dtype=object)
    
    >>> ndexpr_diff([x**2+2*y/z, sin(x)], (x,y))
    array([[2*x, 2/z],
           [cos(x), 0]], dtype=object)
    
    '''
    ndexpr = np.asarray(ndexpr)
    wrt = np.asarray(wrt)
    jac = np.empty(ndexpr.shape + wrt.shape, dtype=object)
    for i, elem in np.ndenumerate(wrt):
        jac[(...,) + i] = ew_diff(ndexpr, elem)
    
    return jac


def flat_cat(*args, **kwargs):
    '''Concatenate flattened arrays.'''
    chain = itertools.chain(args, kwargs.values())
    return np.concatenate([np.asanyarray(a).flatten() for a in chain])

