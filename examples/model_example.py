"""Example of model code generation."""


import functools

import sympy

from sym2num import model, printing, utils, var


class ExampleModel(model.Base):
    
    derivatives = [('df_dx', 'f', 'x'), 
                   ('df_dx_dt', 'df_dx', 't'),
                   ('dg_dx', 'g', 'x')]
    generate_functions = ['f', 'df_dx', 'g', 'dg_dx']
    generate_sparse = ['df_dx', 'f']
    
    @property
    def variables(self):
        """Model variables."""
        v = super().variables
        v['x'] = ['u', 'v', 'V']
        v['t'] = 't'
        v['y'] = [['p'], ['q']]
        return v

    @property
    def member_variables(self):
        """Model member variables."""
        v = super().variables
        v['consts'] = ['M', 'rho', 'h']
        v['T'] = BivariateCallable('T')
        return v
    
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


def reload_all():
    """Reload dependencies for testing"""
    import imp
    for m in (var, printing, model):
        imp.reload(m)


if __name__ == '__main__':
    e = ExampleModel()
    print(model.print_class(e))
