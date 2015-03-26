'''Model generation test.'''


import numpy as np
import sympy

from sym2num import model


class ModelA(model.SymbolicModel):
    t = 't'
    x = ['x1', 'x2']
    y = [['a', 'b'], ['c', 'd']]
    
    var_names = ['t', 'x', 'y']
    function_names = ['f', 'g']
    
    def f(self, t, x):
        s = self.symbols(t, x)
        d = {'x1': s.t * s.x1 ** 2,
             'x2': sympy.exp(s.t * s.x2)}
        return self.pack('x', d)
    
    def g(self, t, y):
        s = self.symbols(t, y)
        return np.dot(y, y.T) * sympy.cos(s.t)
