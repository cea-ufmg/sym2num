"""Models scipy splines for code generation."""


import numbers

import sympy
from sympy.core.function import ArgumentIndexError


class UnivariateSpline(sympy.Function):
    nargs = (1, 2)
    """Number of function arguments."""
    
    @property
    def dx(self):
        return self.args[1] if len(self.args) == 2 else 0
    
    def fdiff(self, argindex=1):
        if argindex == 2:
            raise ValueError("Only derivatives wrt first argument allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        dx = self.dx
        return self.__class__(self.args[0], dx + 1)


class BivariateSpline(sympy.Function):
    nargs = (2, 4)
    """Number of function arguments."""
    
    @property
    def dx(self):
        return self.args[2] if len(self.args) == 4 else 0

    @property
    def dy(self):
        return self.args[3] if len(self.args) == 4 else 0
    
    def fdiff(self, argindex=1):
        if argindex > 2:
            raise ValueError("Only derivatives wrt x and y allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        dx = self.dx + 1 if argindex == 1 else self.dx
        dy = self.dy + 1 if argindex == 2 else self.dy
        return self.__class__(*self.args[:2], dx, dy)


def NamedUnivariateSpline(name):
    metaclass = type(UnivariateSplineBase)
    return metaclass(name, (UnivariateSplineBase,), {})


def NamedBivariateSpline(name):
    metaclass = type(BivariateSplineBase)
    return metaclass(name, (BivariateSplineBase,), {})

