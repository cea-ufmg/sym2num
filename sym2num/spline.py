"""Models scipy splines for code generation."""


import numbers

import sympy
from sympy.core.function import ArgumentIndexError

from . import utils


class SplineBase(sympy.Function):
    """Base class for code generation splines."""
    
    @classmethod
    def print_prepare_validate(cls, printer):
        """Returns code to validate and prepare the variable from arguments."""
        return ''
    
    @utils.classproperty
    def broadcast_elements(cls):
        """List of elements which should be broadcast to generate the output."""
        return []
    
    @classmethod
    def subs_dict(cls, value):
        """Dictionary of substitutions to evaluate with a given value."""
        name = getattr(cls, 'name', None) or cls.__name__
        return {cls: value}
    
    @utils.classproperty
    def identifiers(cls):
        """Set of identifiers defined in this variable's code."""
        return {cls.name}


class UnivariateSplineBase(SplineBase):
    nargs = (1, 2)
    """Number of function arguments."""
    
    @property
    def dx(self):
        if len(self.args) == 1:
            return 0
        assert isinstance(self.args[1], (numbers.Integral, sympy.Integer))
        return self.args[1]
    
    def fdiff(self, argindex=1):
        if argindex == 2:
            raise ValueError("Only derivatives wrt first argument allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        dx = self.dx
        return self.__class__(self.args[0], dx + 1)


class BivariateSplineBase(SplineBase):
    nargs = (2, 4)
    """Number of function arguments."""
    
    @property
    def dx(self):
        if len(self.args) == 2:
            return 0
        assert isinstance(self.args[2], (numbers.Integral, sympy.Integer))
        return self.args[2]
    
    @property
    def dy(self):
        if len(self.args) == 2:
            return 0
        assert isinstance(self.args[3], (numbers.Integral, sympy.Integer))
        return self.args[3]
    
    def fdiff(self, argindex=1):
        if argindex > 2:
            raise ValueError("Only derivatives wrt x and y allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        dx = self.dx + 1 if argindex == 1 else self.dx
        dy = self.dy + 1 if argindex == 2 else self.dy
        return self.__class__(*self.args[:2], dx, dy)


def UnivariateSpline(name):
    metaclass = type(UnivariateSplineBase)
    return metaclass(name, (UnivariateSplineBase,), {'name': name})


def BivariateSpline(name):
    metaclass = type(BivariateSplineBase)
    return metaclass(name, (BivariateSplineBase,), {'name': name})
