"""Sympy functions generate specific code."""


import sympy


class invert(sympy.Function):
    """Sympy function to generate Python's bitwise inversion operator `~`."""

    nargs = (1,)
    
    def __invert__(self):
        return self.args[0]


class getmaskarray(sympy.Function):
    """Sympy function to generate `numpy.ma.getmaskarray`."""

    nargs = (1,)

    def __invert__(self):
        return invert(self)
    
    def numpyrepr(self, printer, *args, **kwargs):
        arg = printer._print(self.args[0])
        return '{}.ma.getmaskarray({})'.format(printer.numpy, arg)

