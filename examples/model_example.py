"""Example of model code generation."""


import sympy

from sym2num import model, utils, var


class ExampleModel(model.Base):
    
    derivatives = [('df_dx', 'f', 'x')]

    @model.make_variables_dict
    def variables():
        """Model variables definition."""
        return [
            var.SymbolArray('x', ['u', 'v']),
            var.SymbolArray('t'),
            var.SymbolArray('y', [['p'], ['q']])
        ]
    
    @model.symbols_from('t, x')
    def f(self, a):
        """Example method."""
        return sympy.Array([a.v, a.t**2 + a.u])
