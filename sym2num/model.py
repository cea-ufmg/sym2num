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
class {{name}}({{inheritance}}):
    """Generated code for symbolic model {{sym_name}}"""
    var_specs = {{{specs}}}
    """Specification of the model variables."""
    
    signatures = {{{signatures}}}
    """Signatures of the model functions."""
    {{#functions}}

    @staticmethod
{{{def}}}
    {{/functions}}
'''

class SymbolicModel(metaclass=abc.ABCMeta):
    symbol_assumptions = {'real': True}

    def __init__(self):
        # Create the model variables
        assumptions = self.symbol_assumptions
        self.vars = {}
        self.var_specs = {}
        for var_name in self.var_names:
            specs = getattr(self, var_name)
            var = np.zeros(np.shape(specs), dtype=object)
            for index, element_name in np.ndenumerate(specs):
                var[index] = sympy.Symbol(element_name, **assumptions)
            self.vars[var_name] = var
            self.var_specs[var_name] = specs
        
        # Create the model functions
        self.functions = {}
        self.signatures = {}
        for fname in self.function_names:
            f = getattr(self, fname)
            if not callable(f):
                raise TypeError('Function `{}` not callable.'.format(fname))
            if isinstance(f, types.MethodType):
                signature = inspect.getfullargspec(f).args[1:]
            else:
                signature = inspect.getfullargspec(f).args
            args = [self.vars[var] for var in signature]
            self.functions[fname] = function.SymbolicFunction(f, args)
            self.signatures[fname] = signature
        
        # Add the derivatives
        for spec in self.derivatives:
            self.add_derivative(*spec)
    
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
    
    def print_class(self, printer, name=None, bases=[], meta=None):
        sym_name = type(self).__name__
        if name is None:
            name = re.sub('Symbolic', 'Generated', sym_name)
        inheritance = list(bases) + (["metaclass=" + meta] if meta else [])
        
        tags = dict(name=name, indent=printing.indent, sym_name=sym_name)
        tags['specs'] = self.var_specs
        tags['signatures'] = self.signatures
        tags['inheritance'] = ', '.join(inheritance)
        tags['functions'] = [{'def': printing.indent(fsym.print_def(printer))}
                             for fsym in self.functions.values()]
        
        return pystache.render(class_template, tags)
    
    def add_derivative(self, name, fname, wrt_names):
        if isinstance(wrt_names, str):
            wrt_names = (wrt_names,)
        
        f = self.functions[fname]
        for wrt_name in reversed(wrt_names):
            f = f.diff(self.vars[wrt_name], name)
        
        self.functions[name] = f
        self.signatures[name] = self.signatures[fname]

