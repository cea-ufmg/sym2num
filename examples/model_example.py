"""Example of model code generation."""


import functools, imp

import sympy

from sym2num import model, printing, utils, var


# Reload dependencies for testing
for m in (var, printing, model):
    imp.reload(m)


class ExampleModel(model.Base):
    
    generate_functions = ['f', 'df_dx', 'g', 'dg_dx']
    generate_sparse = ['df_dx', 'f']
    
    def init_variables(self):
        """Initialize model variables."""
        v = self.variables
        v['x'] = ['u', 'v', 'V']
        v['t'] = 't'
        v['y'] = [['p'], ['q']]
        v['self'] = {
            'consts': ['M', 'rho', 'h'], 
            'T': var.BivariateCallable('T')
        }
    
    def init_derivatives(self):
        """Initialize model derivatives."""
        self.add_derivative('df_dx', 'f', 'x')
        self.add_derivative('df_dx_dt', 'df_dx', 't')
        self.add_derivative('dg_dx', 'g', 'x')
    
    @property
    def generate_assignments(self):
        return dict(nx=len(self.variables['x']),
                    yshape=self.variables['y'].shape)
    
    @model.collect_symbols
    def f(self, t, x, *, a):
        """Example method."""
        return [a.v, a.t**2 + a.u]
    
    @model.collect_symbols
    def g(self, t, x, y, *, a):
        """Another example method."""
        return [a.T(a.V, a.rho)**2]


if __name__ == '__main__':
    e = ExampleModel()
    #print(model.print_class(e))
