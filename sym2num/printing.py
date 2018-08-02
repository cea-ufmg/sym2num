'''Sympy printer for numeric code generation.'''


import re

import numpy as np
import sympy

from sympy.printing.pycode import SciPyPrinter


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

