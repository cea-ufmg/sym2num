'''Symbolic model code generation.'''


import abc
import inspect
import types

import attrdict
import numpy as np
import pystache
import sympy

from . import function, printing


class_template = '''\
class {{name}}({{bases}}):
{{#indent}}

{{/indent}}
'''

class SymbolicModel(metaclass=abc.ABCMeta):
    symbol_assumptions = {'real': True}

    def __init__(self):
        # Create the model variables
        assumptions = self.symbol_assumptions
        self.vars = {}
        for var_name in self.var_names:
            template = getattr(self, var_name)
            var = np.zeros(np.shape(template), dtype=object)
            for index, element_name in np.ndenumerate(template):
                var[index] = sympy.Symbol(element_name, **assumptions)
            self.vars[var_name] = var
        
        # Check for duplicate symbols
        symbols = np.concatenate([a.flatten() for a in self.vars.values()])
        unique_symbols = set(symbols)
        if len(unique_symbols) != len(symbols):
            raise ValueError("Duplicate symbols in model variables.")
        
        # Create the model functions
        self.functions = {}
        self.signatures = {}
        for f_name in self.function_names:
            f = getattr(self, f_name)
            if not callable(f):
                raise TypeError('Function `{}` not callable.'.format(f_name))
            if isinstance(f, types.MethodType):
                signature = inspect.getfullargspec(f).args[1:]
            else:
                signature = inspect.getfullargspec(f).args
            args = [self.vars[var] for var in signature]
            self.functions[f_name] = function.SymbolicFunction(f(*args), args)
            self.signatures[f_name] = signature
    
    @property
    @abc.abstractmethod
    def var_names(self):
        '''List of the model variable names.'''
        raise NotImplementedError("Pure abstract method.")

    @property
    @abc.abstractmethod
    def function_names(self):
        '''List of the model function names.'''
        raise NotImplementedError("Pure abstract method.")

    def pack(self, name, d):
        var = self.vars[name]
        ret = np.zeros(var.shape, dtype=object)
        for index, elem in np.ndenumerate(var):
            try:
                ret[index] = d[elem.name]
            except KeyError:
                pass
        return ret

    @staticfunction
    def symbols(*args):
        symbol_list = np.concatenate([np.flatten(a) for a in args])
        return attrdict.AttrDict({s.name: s for s in symbol_list})

    def print_class(self, name, printer, bases=[]):
        tags = dict(name=name, indent=printing.indent)
        tags['bases'] = ', '.join(bases)
        return pystache.render(class_template, tags)
