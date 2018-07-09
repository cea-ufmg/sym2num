"""Example of model code generation."""


import sympy

from sym2num import model, utils, var


class ExampleModel:
    @utils.init_static_variable
    def variables():
        """Model Variables."""
        x = var.SymbolArray(['u', 'v'])
        t = var.SymbolArray('t')
        y = var.SymbolArray(
            [[sympy.Symbol('p', complex=True)], 
             [sympy.Symbol('q', complex=True)]]
        )
        return dict(x=x, t=t, y=y)
    
    @model.symbols_from('t, x')
    def f(self, a):
        """Example method."""
        return sympy.Array([a.v, a.t**2 + a.u])
