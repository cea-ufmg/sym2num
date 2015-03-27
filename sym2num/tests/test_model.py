'''Model generation test.'''


import numpy as np
import pytest
import sympy

from sym2num import model, printing


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


@pytest.fixture(scope="module")
def a():
    return ModelA()


@pytest.fixture(scope="module")
def generated(a, printer):
    code = '\n'.join(printer.imports + (a.print_class(printer, 'A'),))
    context = {}
    exec(code, context)
    A = context['A']
    return A()


def test_generated(a, generated, seed):
    '''Test the generated model functions against their symbolic expression.'''
    subs = {}
    vars = {}
    for var_name, var in a.vars.items():
        value = np.random.standard_normal(var.shape)
        vars[var_name] = value
        for index, elem in np.ndenumerate(var):
            subs[elem] = value[index]
    
    for name, symfun in a.functions.items():
        sig = a.signatures[name]
        num = getattr(generated, name)(*[vars[name] for name in sig])
        for index, value in np.ndenumerate(num):
            assert abs(value - symfun.out[index].subs(subs)) < 1e-9

