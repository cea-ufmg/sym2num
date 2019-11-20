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
    
    from sympy.abc import t, x, y
    output = sympy.Array(
        [x**2 + sympy.erf(x),
         sympy.cos(y) + 2*t + sympy.GoldenRatio]
    )
    arguments = function.Arguments(t=t, state=[x, y])
    
    f = function.FunctionPrinter('f', output, arguments)
    print(f.print_def())
