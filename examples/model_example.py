"""Example of model code generation."""


import sympy

from sym2num import model, spline, utils, var


import imp
[imp.reload(m) for m in (utils, spline, var, model)]

class ExampleModel(model.Base):
    
    derivatives = [('df_dx', 'f', 'x'), ('df_dx_dt', 'df_dx', 't')]
    generate_functions = ['f', 'df_dx', 'g']
    generate_sparse = ['df_dx', 'f']
    
    @model.make_variables_dict
    def variables():
        """Model variables definition."""
        return [
            var.SymbolObject('self', 
                             var.SymbolArray('consts', ['M', 'rho', 'h']), 
                             spline.BivariateSpline('T')),
            var.SymbolArray('x', ['u', 'v', 'V']),
            var.SymbolArray('t'),
            var.SymbolArray('y', [['p'], ['q']])
        ]
    

    @property
    def generate_assignments(self):
        return dict(nx=len(self.variables['x']),
                    yshape=self.variables['y'].shape)
    
    @model.symbols_from('t, x')
    def f(self, a):
        """Example method."""
        return sympy.Array([a.v, a.t**2 + a.u])
    
    @model.symbols_from('t, x, y')
    def g(self, a):
        """Another example method."""
        return sympy.Array([self.T(a.V, a.p)**2])


if __name__ == '__main__':
    e = ExampleModel()
    print(model.print_class("ExampleModel", e))
