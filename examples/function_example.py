import collections

import numpy as np
import sympy

from sym2num import function, var


def reload_all():
    """Reload modules for testing."""
    import imp
    for m in (var, function):
        imp.reload(m)


if __name__ == '__main__':
    reload_all()
    
    g = var.UnivariateCallable('g')
    h = var.UnivariateCallable('h')

    from sympy.abc import t, w, x, y, z, m
    output = [x**2 + sympy.erf(x) + g(x),
              sympy.cos(y) + 2*t + sympy.GoldenRatio,
              z*sympy.sqrt(sympy.sin(w)+2)*h(x, 2)]
    obj = {'data': [w], 'extra': {'other': [m, z]}, 'gg': g}
    arguments = function.Arguments(self=obj, t=t, state=[x, y], H=h)
    
    f = function.FunctionPrinter('f', output, arguments)
    print(f.print_def())

    
