"""Example of model code generation."""


import functools, imp

import numpy as np
import sympy

from sym2num import model, function, printing, utils, var


# Reload dependencies for testing
for m in (var, printing, function, model):
    imp.reload(m)


class ExampleModel(model.Base):
    
    generate_functions = ['f', 'df_dx', 'g', 'C']
    
    def init_variables(self):
        """Initialize model variables."""
        v = self.variables
        v['x'] = ['u', 'v', 'V']
        v['t'] = 't'
        v['y'] = [['p'], ['q']]
        
        v['self']['consts'] = ['M', 'rho', 'h']
        v['self']['T'] = var.BivariateCallable('T')
    
    def init_derivatives(self):
        """Initialize model derivatives."""
        self.add_derivative('f', 'x')
        self.add_derivative('f', ('x', 't'))
        self.add_derivative('g', 'x', deriv_name='C')
    
    @property
    def generate_assignments(self):
        return dict(nx=len(self.variables['x']),
                    yshape=self.variables['y'].shape,
                    array_test=np.array([3,2,1]))
    
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
    print(model.print_class(e))
