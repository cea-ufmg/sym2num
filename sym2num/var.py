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
        return {s.name for s in self.symbols}
    
    @property
    def symbols(self):
        """Set of symbols defined by this variable."""
        raise NotImplementedError('must be implemented by subclasses')
    
    def subs_map(self, value):
        """Create mapping of this variable's symbols to the value given."""
        raise NotImplementedError('must be implemented by subclasses')


class SymbolObject(Variable, collections.OrderedDict):
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f'no attribute named {name}') from None

    def __setitem__(self, key, item):
        if not utils.isstr(key):
            raise TypeError(f'key should be of class `str`')
        if not utils.isidentifier(key):
            raise ValueError(f'key "{key}" is not a valid identifier')
        v = variable(item)
        super().__setitem__(key, v)
    
    @property
    def symbols(self):
        """Set of symbols defined by this variable."""
        s = set()
        for v in self.values():
            s |= v.symbols
        return s
    
    def ndenumerate(self):
        """ndenumeration of this object SymbolArrays"""
        for name, var in self.items():
            if isinstance(var, SymbolArray):
                for ind, symbol in var.ndenumerate():
                    yield name, ind, symbol
            elif isinstance(var, SymbolObject):
                for attrname, ind, symbol in var.ndenumerate():
                    yield f'{name}.{attrname}', ind, symbol
    
    def callables(self):
        """ndenumeration of this object Callables"""
        for name, var in self.items():
            if isinstance(var, type) and issubclass(var, CallableBase):
                yield name, var.name
            elif isinstance(var, SymbolObject):
                for attrname, varname in var.callables():
                    yield f'{name}.{attrname}', varname
    
    def subs_map(self, value):
        """Create mapping of this variable's symbols to the value given."""
        ret = {}
        for attrname, attrvar in self.items():
            attrvalue = getattr(value, attrname)
            ret.update(attrvar.subs_map(attrvalue))
        return ret


class SymbolArray(Variable, np.ndarray):
    """Represents array of symbols for code generation."""
    
    def __new__(cls, spec, gen_dtype='float_'):
        if isinstance(spec, sympy.Symbol):
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

        if len(set(arr.flat)) != arr.size:
            raise ValueError('repeated values in symbol array')
        
        obj = super().__new__(cls, arr.shape, dtype=object)
        obj[...] = arr        
        obj.gen_dtype = gen_dtype
        return obj
    
    def ndenumerate(self):
        yield from np.ndenumerate(self)
    
    @property
    def symbols(self):
        """Set of symbols defined by this variable."""
        return {elem for elem in self.flat}
    
    def subs_map(self, value):
        """Create mapping of this variable's symbols to the value given."""
        value_array = np.asarray(value, object)
        if self.shape != value_array.shape:
            raise ValueError('invalid shape for argument')
        
        return {self[i]: e for i,e in np.ndenumerate(value_array)}


class CallableMeta(Variable, sympy.FunctionClass):
    """Metaclass of code generation callables."""
    
    @property
    def symbols(self):
        """Set of symbols defined by this variable."""
        return {self}
    
    @property
    def name(self):
        """Name of this variable."""
        return self.__name__
    
    def subs_map(self, value):
        """Create mapping of this variable's symbols to the value given."""
        return {self: value}


class CallableBase:
    """Base class for code-generation callables like in `scipy.interpolate`."""
    
    @utils.classproperty
    def fname(cls):
        """Name of the function."""
        return cls.name


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
    d = {'nargs': (1,2)}
    return CallableMeta(name, (UnivariateCallableBase, sympy.Function), d)


def BivariateCallable(name):
    metaclass = type(sympy.Function)
    d = {'nargs': (2, 4)}
    return CallableMeta(name, (BivariateCallableBase, sympy.Function), d)


def variable(spec):
    """Make a symbolic variable from simple specifications."""
    if isinstance(spec, Variable):
        return spec
    elif isinstance(spec, (str, sympy.Symbol, list, np.ndarray)):
        return SymbolArray(spec)
    elif isinstance(spec, dict):
        return SymbolObject(spec)
    else:
        raise TypeError("unrecognized variable specification type")
