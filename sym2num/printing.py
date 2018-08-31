'''Sympy printer for numeric code generation.'''


import re

import numpy as np
import sympy

from sympy.printing.pycode import SciPyPrinter

from . import utils


class Printer(SciPyPrinter):
    """sym2num sympy code printer."""
    
    import_aliases = {
        'numpy': '_np',
        'scipy': '_scipy',
        'scipy.special': '_scipy_special',
        'scipy.constants': '_scipy_constants',
        'scipy.sparse': '_scipy_sparse'
    }
    
    @property
    def numpy_alias(self):
        return self.import_aliases.get('numpy', 'numpy')
    
    def _module_format(self, fqn, register=True):
        super()._module_format(fqn, register)
        parts = fqn.split('.')
        module = '.'.join(parts[:-1])
        try:
            alias = self.import_aliases[module]
            return str.join('.', (alias, parts[-1]))
        except KeyError:
            return fqn
    
    @property
    def direct_imports(self):
        return (m for m in self.module_imports if m not in self.import_aliases)
    
    @property
    def aliased_imports(self):
        for module, alias in self.import_aliases.items():
            if module in self.module_imports:
                yield module, alias

    def print_ndarray(self, arr, assign_to=None):
        arr = np.asarray(arr)
        subs = dict(
            np=self.numpy_alias,
            dtype=arr.dtype,
            list=arr.tolist(),
            shape=arr.shape
        )
        if arr.size:
            arr_str = "{np}.array({list}, dtype={np}.{dtype})".format(**subs)
        else:
            arr_str = "{np}.zeros({shape}, dtype={np}.{dtype})".format(**subs)

        if assign_to and utils.isidentifier(assign_to):
            return '{} = {}'.format(assign_to, arr_str)
        else:
            return arr_str

    def _print_UnivariateSplineBase(self, expr):
        x = self._print(expr.args[0])
        if expr.dx == 0:
            return f'{expr.__class__.__name__}({x})'
        else:
            return f'{expr.__class__.__name__}({x}, dx={expr.dx})'
        
    def _print_BivariateSplineBase(self, expr):
        x = self._print(expr.args[0])
        y = self._print(expr.args[1])
        if expr.dx == 0 and expr.dy == 0:
            return f'{expr.__class__.__name__}({x}, {y})'
        else:
            args = f'{x}, {y}, dx={expr.dx}, dy={expr.dy}'
            return f'{expr.__class__.__name__}({args})'
