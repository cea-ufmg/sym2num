"""Example of model code generation."""


import sympy

from sym2num import model, utils, var


class ExampleModel:
    @utils.init_static_variable
    def variables():
        """Model Variables."""
        return [
            var.SymbolArray('x', ['u', 'v']),
            var.SymbolArray('t'),
            var.SymbolArray('y', [['p'], ['q']])
        ]
    
    @model.symbols_from('t, x')
    def f(self, a):
        """Example method."""
        return sympy.Array([a.v, a.t**2 + a.u])
