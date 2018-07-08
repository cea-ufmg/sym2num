"""Example of model code generation."""


import sympy

from sym2num import utils, var

def atoms_from(*args, **kwargs):
    pass

class ExampleModel:
    @utils.init_static_variable
    def variables():
        """Model Variables."""
        x = var.SymbolArray(['u', 'v'])
        t = var.SymbolArray('t')
        y = sympy.SymbolArray(
            [[sympy.Symbol('p', complex=True)], 
             [sympy.Symbol('q', complex=True)]]
        )
        return dict(x=x, t=t, y=y)
    
    @atoms_from('t, x')
    @staticmethod
    def f(a):
        """Example method."""
        return sympy.Array([a.v, a.t**2 + a.u])
