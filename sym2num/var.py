"""Symbolic variables for code generation.

We consider three kinds of variables:

* sympy.Symbol ndarrays, specified as  (nested) lists of str, for convenience.
* Custom callables.
* Objects of the previous variables, specified as dicts

"""


import collections
import inspect
import keyword
import numbers
import re

import attrdict
import jinja2
import numpy as np
import sympy
from sympy.core.function import ArgumentIndexError

from . import utils


class Variable:
    """Base class of code generation variables."""
    
    @property
    def identifiers(self):
        """Set of symbol identifiers defined by this variable."""
        raise NotImplementedError('must be implemented by subclasses')


class SymbolObject(Variable, dict):
    def __init__(self, spec):
        for key, val in spec.items():
            assert utils.isidentifier(key)
            self[key] = variable(val)

    @property
    def identifiers(self):
        """Set of symbol identifiers defined by this variable."""
        s = set()
        for v in self.values():
            s |= v.identifiers
        return s


class SymbolArray(Variable):
    """Represents array of symbols for code generation."""
    
    def __init__(self, spec):
        if isinstance(spec, sympy.Symbol)
            arr = np.asarray(spec, object)
        elif utils.isstr(spec):
            arr = np.asarray(sympy.Symbol(spec), object)
        elif isinstance(spec, list):
            names = np.asarray(spec, str)
            arr = np.empty(names.shape, object)
            for ind, name in np.ndenumerate(names):
                assert utils.isidentifier(name)
                arr[ind] = sympy.Symbol(name)
        elif isinstance(spec, np.ndarray):
            arr = spec
        else:
            raise TypeError("unrecognized variable specification type")
                
        self.arr = arr
        """Underlying symbol ndarray."""
        
        if len(self.identifiers) != self.arr.size:
            raise ValueError('repeated values in symbol array')
    
    @property
    def identifiers(self):
        """Set of symbol identifiers defined by this variable."""
        return {elem.name for elem in self.arr.flat}


class CallableBase(Variable):
    """Base class for code-generation callables like in `scipy.interpolate`."""
    
    @utils.classproperty
    def symbols(cls):
        """Set of symbols defined by this variable."""
        return {cls.name}

    @utils.classproperty
    def name(cls):
        """Name of this variable."""
        return cls.__name__


class UnivariateCallableBase(CallableBase):
    """Base for univariate callables like Pchip, PPoly, Akima1d, Spline, etc."""
    
    def __init__(self, *args):
        if len(args) == 2:
            if not isinstance(self.args[1], (numbers.Integral, sympy.Integer)):
                raise TypeError('derivative arguments must be integers')
    
    @property
    def dx(self):
        if len(self.args) == 1:
            return 0
        return self.args[1]
    
    def fdiff(self, argindex=1):
        if argindex == 2:
            raise ValueError("Only derivatives wrt first argument allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        dx = self.dx
        return self.__class__(self.args[0], dx + 1)


class BivariateCallableBase(CallableBase):
    """Base for bivariate callables like scipy's BivariateSpline."""
    
    def __init__(self, *args):
        if len(args) == 4:
            integer = (numbers.Integral, sympy.Integer)
            if (not isinstance(self.args[2], integer)
                or not isinstance(self.args[3], integer)):
                raise TypeError('derivative arguments must be integers')
    
    @property
    def dx(self):
        if len(self.args) == 2:
            return 0
        return self.args[2]
    
    @property
    def dy(self):
        if len(self.args) == 2:
            return 0
        return self.args[3]
    
    def fdiff(self, argindex=1):
        if argindex > 2:
            raise ValueError("Only derivatives wrt x and y allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        dx = self.dx + 1 if argindex == 1 else self.dx
        dy = self.dy + 1 if argindex == 2 else self.dy
        return self.__class__(*self.args[:2], dx, dy)


def UnivariateCallable(name):
    metaclass = type(sympy.Function)
    d = {'nargs': (1,2)}
    return metaclass(name, (UnivariateCallableBase, sympy.Function), d)


def BivariateCallable(name):
    metaclass = type(sympy.Function)
    d = {'nargs': (2, 4)}
    return metaclass(name, (BivariateCallableBase, sympy.Function), d)


def variable(spec):
    """Make a symbolic variable from simple specifications."""
    if isinstance(spec, (str, list, np.ndarray)):
        return SymbolArray(spec)
    elif isinstance(spec, CallableBase):
        return spec
    elif isinstance(spec, dict):
        return SymbolObject(spec)
    else:
        raise TypeError("unrecognized variable specification type")


class Dict(collections.OrderedDict):
    """Dictionary of code generation variables."""
    
    def __setitem__(self, key, item):
        if not utils.isstr(key):
            raise TypeError(f'key should be of class `str`')
        if not utils.isidentifier(key):
            raise ValueError(f'key "{key}" is not a valid identifier')
        v = variable(item)
        super().__setitem__(key, v)
