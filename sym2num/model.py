'''Symbolic model code generation.

Improvement ideas
-----------------
 * Use jinja2 instead of pystache.

'''


import abc
import collections.abc
import inspect
import re
import types

import attrdict
import numpy as np
import pystache
import sympy

from . import function, printing, utils


class_template = '''\
class {{name}}({{signature}}):
    """Generated code for symbolic model {{sym_name}}"""

    var_specs = {{{specs}}}
    """Specification of the model variables."""    
    {{#functions}}

    @staticmethod
{{{def}}}
    {{/functions}}
'''


parametrized_call_template = '''\
def {fname}(self, {signature}):
    """Parametrized version of `{fname}`."""
    return self.call(_wrapped_function, {args})
'''


class SymbolicModel(metaclass=abc.ABCMeta):
    symbol_assumptions = {'real': True}

    def __init__(self):
        self._init_variables()
        self._init_functions()
        self._init_derivatives()
    
    def _init_variables(self):
        '''Initialize the model variables.'''
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
    
    def _init_functions(self):
        '''Initialize the model functions.'''
        self.functions = {}
        for fname in self.function_names:
            f = getattr(self, fname)
            if not callable(f):
                raise TypeError('Function `{}` not callable.'.format(fname))
            if isinstance(f, types.MethodType):
                argnames = inspect.getfullargspec(f).args[1:]
            else:
                argnames = inspect.getfullargspec(f).args
            args = [(name, self.vars[name]) for name in argnames]
            self.functions[fname] = function.SymbolicFunction(f, args)
    
    def _init_derivatives(self):
        '''Initialize model derivatives.'''
        for spec in getattr(self, 'derivatives', []):
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

    def symbols(self, *args, **kwargs):
        symbol_list = utils.flat_cat(*args)
        symbols = attrdict.AttrDict({s.name: s for s in symbol_list})
        for argname, value in kwargs.items():
            var = self.vars[argname]
            for i, xi in np.ndenumerate(value):
                symbols[var[i].name] = xi
        return symbols
    
    def print_class(self, printer, name=None, signature=''):
        sym_name = type(self).__name__
        if name is None:
            name = sym_name
        
        tags = dict(name=name, specs=self.var_specs, sym_name=sym_name)
        tags['signature'] = signature
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


def class_obj(sym, printer, context=None, name=None, meta=None):
    # Get the default arguments if none were given
    if name is None:
        name = type(sym).__name__
    if context is None:
        context = {}
    if meta is None:
        signature = ''
    else:
        signature = 'metaclass=_generation_meta'
        context['_generation_meta'] = meta
    
    # Load the printer imports and execute the model class code
    imports = '\n'.join(printer.imports)
    class_code = sym.print_class(printer, name=name, signature=signature)
    exec(imports, context)
    exec(class_code, context)
    return context[name]


class ParametrizedModel:
    def __init__(self, params={}):
        # Save a copy of the given params
        self._params = {k: np.asarray(v) for k, v in params.items()}
        
        # Add default parameters for the empty variables
        for name, spec in self.var_specs.items():
            if np.size(spec) == 0 and name not in params:
                self._params[name] = np.array([])
    
    def parametrize(self, params={}, **kwparams):
        '''Parametrize a new class instance with the given + current params.'''
        new_params = self._params.copy()
        new_params.update(params)
        new_params.update(kwparams)
        return type(self)(new_params)
    
    def call_args(self, f, *args, **kwargs):
        fargs = inspect.getfullargspec(f).args
        call_args = {k: v for k, v in self._params.items() if k in fargs}
        call_args.update(filterout_none(kwargs))
        call_args.update(filterout_none(zip(fargs, args)))
        return call_args
    
    def call(self, f, *args, **kwargs):
        call_args = self.call_args(f, *args, **kwargs)
        return f(**call_args)
    
    @staticmethod
    def decorate(f):
        args = inspect.getfullargspec(f).args
        tags = {'fname': f.__name__, 
                'signature': ', '.join('%s=None' % a for a in args),
                'args': ', '.join(args)}
        context = dict(_wrapped_function=f)
        exec(parametrized_call_template.format(**tags), context)
        return context[f.__name__]
    
    @classmethod
    def meta(cls, name, bases, classdict):
        # Add ourselves to the bases
        if cls not in bases:
            bases = bases + (cls,)
        
        # Decorate the model functions
        for k, v in classdict.items():
            if isinstance(v, staticmethod):
                classdict[k] = cls.decorate(v.__func__)
        
        # Return the new class type
        return type(name, bases, classdict)

    @classmethod
    def pack(cls, name, d):
        spec = np.array(cls.var_specs[name])
        ret = np.zeros(spec.shape)
        for index, elem_name in np.ndenumerate(spec):
            try:
                ret[index] = d[elem_name]
            except KeyError:
                pass
        return ret


def filterout_none(d):
    '''Returns a mapping without the values which are `None`.'''
    items = d.items() if isinstance(d, collections.abc.Mapping) else d
    return {k: v for k, v in items if v is not None}

