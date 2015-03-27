'''Symbolic model code generation.'''


import abc
import inspect
import re
import types

import attrdict
import numpy as np
import pystache
import sympy

from . import function, printing, utils


class_template = '''\
class {{name}}({{bases}}):
    """Generated code for symbolic model {{sym_name}}"""
    {{#functions}}

    @staticmethod
{{def}}
    {{/functions}}
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
        symbols = utils.flat_cat(**self.vars)
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
            self.functions[f_name] = function.SymbolicFunction(f, args)
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

    @staticmethod
    def symbols(*args, **kwargs):
        symbol_list = utils.flat_cat(*args, **kwargs)
        return attrdict.AttrDict({s.name: s for s in symbol_list})

    def print_class(self, printer, name=None, bases=[]):
        sym_name = type(self).__name__
        if name is None:
            name = re.sub('Symbolic', 'Generated', sym_name)
        
        tags = dict(name=name, indent=printing.indent, sym_name=sym_name)
        tags['bases'] = ', '.join(bases)
        tags['functions'] = [{'def': printing.indent(fsym.print_def(printer))}
                             for fname, fsym in self.functions.items()]
        
        return pystache.render(class_template, tags)
