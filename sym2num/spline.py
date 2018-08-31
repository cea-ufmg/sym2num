"""Models scipy splines for code generation."""


import numbers

import sympy
from sympy.core.function import ArgumentIndexError


class UnivariateSplineBase(sympy.Function):
    nargs = (1, 2)
    """Number of function arguments."""
    
    def fdiff(self, argindex=1):
        if argindex == 2:
            raise ValueError("Only derivatives wrt first argument allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        x, dx = self.args if len(self.args) == 2 else (*self.args, 0)
        return self.__class__(x, dx + 1)


class BivariateSplineBase(sympy.Function):
    nargs = (2, 4)
    """Number of function arguments."""
    
    def __eq__(self, other):
        return (super().__eq__(other) and other.dx == self.dx 
                and other.dy == self.dy)
    
    def fdiff(self, argindex=1):
        if argindex > 2:
            raise ValueError("Only derivatives wrt x and y allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        x,y,dx,dy = self.args if len(self.args) == 4 else (*self.args, 0, 0)
        dx = dx + 1 if argindex == 1 else dx 
        dy = dy + 1 if argindex == 2 else dy        
        return self.__class__(x, y, dx, dy)


def UnivariateSpline(name):
    metaclass = type(UnivariateSplineBase)
    return metaclass(name, (UnivariateSplineBase,), {})


def BivariateSpline(name):
    metaclass = type(BivariateSplineBase)
    return metaclass(name, (BivariateSplineBase,), {})

