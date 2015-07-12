'''Model generation test.'''


import inspect

import numpy as np
import pytest
import sympy

from sym2num import model, printing


class ModelA(model.SymbolicModel):
    var_names = {'t', 'x', 'y'}
    '''Names of the model variables.'''

    function_names = {'f', 'g'}
    '''Names of the model functions.'''

    derivatives = [('df_dx', 'f', 'x'),
                   ('d2f_dx2', 'df_dx', 'x'),
                   ('d2f_dt2', 'f', ('t', 't'))]
    '''List of the model function derivatives to calculate / generate.'''
    
    t = 't'
    '''Specification of the model variable `t`.'''
    
    x = ['x1', 'x2']
    '''Specification of the model variable `x`.'''

    y = [['a', 'b'], ['c', 'd']]
    '''Specification of the model variable `y`.'''

    def f(self, t, x):
        '''Model function `f`.'''
        s = self.symbols(t, x)
        d = {'x1': s.t * s.x1 ** 2,
             'x2': sympy.exp(s.t * s.x2)}
        return self.pack('x', d)
    
    def g(self, t, y):
        '''Model function `g`.'''
        s = self.symbols(t, y)
        return np.dot(y, y.T) * sympy.cos(s.t)


@pytest.fixture(params=range(3))
def seed(request):
    '''Random number generator seed.'''
    np.random.seed(request.param)
    return request.param


@pytest.fixture(scope="module", params=['scipy', 'numpy'])
def printer(request):
    if request.param == 'numpy':
        return printing.NumpyPrinter()
    elif request.param == 'scipy':
        return printing.ScipyPrinter()


def test_generated(printer, seed):
    '''Test the generated model functions against their symbolic expression.'''
    # Instantiate symbolic model
    a = ModelA()
    
    # Generate code and instantiate generated model
    A = model.class_obj(a, printer)
    generated = A()
    
    # Create symbolic substitution tags and numerical values for model variables
    subs = {}
    vars = {}
    for var_name, var in a.vars.items():
        value = np.random.standard_normal(var.shape)
        vars[var_name] = value
        for index, elem in np.ndenumerate(var):
            subs[elem] = value[index]
    
    # Compare the generated and symbolic functions
    for name, symfun in a.functions.items():
        signature = symfun.args.keys()
        num = getattr(generated, name)(*[vars[name] for name in signature])
        for index, value in np.ndenumerate(num):
            assert abs(value - symfun.out[index].subs(subs)) < 1e-9

