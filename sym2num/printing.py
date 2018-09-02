'''Sympy printer for numeric code generation.'''


import re

import numpy as np
import sympy

from sympy.printing.pycode import SciPyPrinter

from . import spline, utils


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

    def _print(self, e):
        # Override print subsystem to prevent collisions of spline names and
        # standard functions like 'gamma' or 'exp'
        if isinstance(e, spline.UnivariateSplineBase):
            return self._print_UnivariateSplineBase(e)
        elif isinstance(e, spline.BivariateSplineBase):
            return self._print_BivariateSplineBase(e)
        else:
            return super()._print(e)
    
    def _print_UnivariateSplineBase(self, e):
        x = self._print(e.args[0])
        name = getattr(e, 'name', None) or e.__class__.__name__
        if e.dx == 0:
            return f'{name}({x})'
        else:
            return f'{name}({x}, dx={e.dx})'
        
    def _print_BivariateSplineBase(self, e):
        x = self._print(e.args[0])
        y = self._print(e.args[1])
        name = getattr(e, 'name', None) or e.__class__.__name__
        if e.dx == 0 and e.dy == 0:
            return f'{name}({x}, {y})'
        else:
            return f'{name}({x}, {y}, dx={e.dx}, dy={e.dy})'
