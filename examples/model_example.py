"""Example of model code generation."""


import functools

import sympy

from sym2num import model, printing, utils, var


import imp
[imp.reload(m) for m in [var, printing, model]]

class ExampleModel(model.Base):
    
    derivatives = [('df_dx', 'f', 'x'), 
                   ('df_dx_dt', 'df_dx', 't'),
                   ('dg_dx', 'g', 'x')]
    generate_functions = ['f', 'df_dx', 'g', 'dg_dx']
    generate_sparse = ['df_dx', 'f']
    
    @utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        vars = [var.SymbolObject('self', 
                                 var.SymbolArray('consts', ['M', 'rho', 'h']),
                                 var.BivariateCallable('T')),
                var.SymbolArray('x', ['u', 'v', 'V']),
                var.SymbolArray('t'),
                var.SymbolArray('y', [['p'], ['q']])]
        return var.make_dict(vars)
    
    @property
    def generate_assignments(self):
        return dict(nx=len(self.variables['x']),
                    yshape=self.variables['y'].shape)
    
    @model.collect_symbols
    def f(self, t, x, *, a):
        """Example method."""
        return sympy.Array([a.v, a.t**2 + a.u])
    
    @model.collect_symbols
    def g(self, t, x, y, *, a):
        """Another example method."""
        return sympy.Array([a.T(a.V, a.rho)**2])


if __name__ == '__main__':
    e = ExampleModel()
    print(model.print_class(e))
