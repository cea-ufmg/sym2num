'''Sympy printer for numeric code generation.'''


import re

import numpy as np
import sympy

from sympy.printing.pycode import SciPyPrinter

from . import utils, var


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
        # Override print subsystem to prevent collisions of custom callable
        # names and standard functions like 'gamma' or 'exp'
        if isinstance(e, var.CallableBase):
            return self._print_CallableBase(e)
        else:
            return super()._print(e)
    
    def _print_CallableBase(self, e):
        args = ', '.join(self._print(arg) for arg in e.args)
        name = getattr(e, 'name', None) or e.__class__.__name__
        return f'{name}({args})'

    def _print_invert(self, e):
        arg = self._print(e.args[0])
        return f'~{arg}'
    
    def _print_getmaskarray(self, e):
        arg = self._print(e.args[0])
        np = self.numpy_alias
        return f'{np}.ma.getmaskarray({arg})'
