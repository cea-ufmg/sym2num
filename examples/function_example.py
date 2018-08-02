import collections

import numpy as np
import sympy

from sym2num import function, var


if __name__ == '__main__':
    from sympy.abc import t, x, y
    output = sympy.Array(
        [x**2 + sympy.erf(x),
         sympy.cos(y) + 2*t + sympy.GoldenRatio]
    )

    arguments = [var.SymbolArray('time', 't'),
                 var.SymbolArray('state', [x, y])]
    
    f = function.NumpyFunction(output, 'f', arguments)
    print(f.code())
