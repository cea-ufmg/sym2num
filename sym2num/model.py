"""Symbolic model code generation.

Improvement ideas
-----------------
* Add compiled code to linecache so that tracebacks can be produced, like done
  in the `IPython.core.compilerop` module.

"""


import abc
import collections
import collections.abc
import functools
import inspect
import itertools
import re
import types

import attrdict
import numpy as np
import jinja2
import sympy

from . import function, printing, utils, var


class Base:
    def __init__(self):
        self._init_derivatives()

    def _init_derivatives(self):
        """Initialize model derivatives."""
        for spec in getattr(self, 'derivatives', []):
            self.add_derivative(*spec)
    
    def __getattr__(self, name):
        assert name != 'variables' # Otherwise we fall in an infinite loop
        
        try:
            self_var = self.variables['self']
        except KeyError:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)
        return getattr(self_var, name)
    
    def function_codegen_arguments(self, fname):
        """Function argument specifications for code generation."""
        try:
            f = getattr(self, fname)
        except AttributeError:
            raise ValueError(f"{fname} member not found")
        except TypeError:
            raise TypeError("function name must be a string")
        
        if isinstance(f, function.SymbolicSubsFunction):
            return f.arguments
        
        param_names = inspect.signature(f).parameters.keys()
        if 'self' in self.variables:
            param_names = itertools.chain(['self'], param_names)
        return [self.variables[n] for n in param_names]
    
    @functools.lru_cache()
    def default_function_output(self, fname):
        """Function output for the default arguments."""
        try:
            f = getattr(self, fname)
        except AttributeError:
            raise ValueError("{} member not found".format(fname))
        except TypeError:
            raise TypeError("function name must be a string")

        if isinstance(f, function.SymbolicSubsFunction):
            return f.output
        
        args = self.function_codegen_arguments(fname)
        if len(args) > 0 and args[0].name == 'self':
            args = args[1:]
        return f(*args)
    
    def add_derivative(self, name, fname, wrt_names):
        if isinstance(wrt_names, str):
            wrt_names = (wrt_names,)
        
        out = self.default_function_output(fname)
        for wrt_name in wrt_names:
            wrt = self.variables[wrt_name]
            out = sympy.derive_by_array(out, wrt)
        
        args = self.function_codegen_arguments(fname)
        deriv = function.SymbolicSubsFunction(args, out)
        setattr(self, name, deriv)


def print_class(model, **options):
    model_printer = ModelPrinter(model, **options)
    return model_printer.print_class()


def compile_class(model, **options):
    model_printer = ModelPrinter(model, **options)
    return model_printer.class_obj()


model_template_src = '''\
# Model imports
import numpy as {{printer.numpy_alias}}

class {{m.name}}:
    """Generated code for {{m.name}} from symbolic model."""
    {% for method in m.methods %}
    {{ method | indent }}
    {% endfor %}
    {% for name, indices in m.sparse_indices.items() -%}
    {{ printer.print_ndarray(indices, assign_to=name) }}
    {% endfor %}
    {% for name, value in m.assignments.items() -%}
    {{ name }} = {{ value }}
    {% endfor %}
'''


class ModelPrinter:
    """Generates numpy code for symbolic models."""

    @utils.cached_class_property
    def template(cls):
        return jinja2.Template(model_template_src)
    
    def __init__(self, model, **options):
        name = (getattr(model, 'generated_name', None)
                or options.get('name', None) 
                or f'Generated{type(model).__name__}')
        self.name = name
        """Name of the generated class."""
        
        self.model = model
        """The underlying symbolic model."""

        try:
            assignments = options['assignments']
        except KeyError:
            assignments = getattr(model, 'generate_assignments', {})
        self.assignments = assignments
        """Mapping of simple assignments to be made in the class code."""
        
        try:
            functions = options['functions']
        except KeyError:
            functions = getattr(model, 'generate_functions', [])

        try:
            sparse = options['sparse']
        except KeyError:
            sparse = getattr(model, 'generate_sparse', [])
        
        f_specs = []
        for fname in functions:
            output = self.model.default_function_output(fname)
            arguments = self.model.function_codegen_arguments(fname)
            f_specs.append((fname, output, arguments))
        
        sparse_indices = collections.OrderedDict()
        for spec in sparse:
            fname, selector = (spec, None) if utils.isstr(spec) else spec
            output = model.default_function_output(fname)
            arguments = model.function_codegen_arguments(fname)
            values, indices = utils.sparsify(output, selector)
            f_specs.append((fname + '_val', values, arguments))
            sparse_indices[fname + '_ind'] = indices
        
        self._f_specs = f_specs
        """Function generation specifications."""

        self.sparse_indices = sparse_indices
        """Indices of sparse functions."""
        
    @property
    def methods(self):
        for fname, output, arguments in self._f_specs:
            fdef = function.print_function(fname, output, arguments)
            if function.isstatic(arguments):
                yield '\n'.join(('@staticmethod', fdef))
            elif function.isclassmethod(arguments):
                yield '\n'.join(('@classmethod', fdef))
            else:
                yield fdef

    def print_class(self):
        context = dict(m=self, printer=printing.Printer())
        return self.template.render(context)

    def class_obj(self):
        env = {}
        exec(compile(self.print_class(), '<string>', 'exec'), env)
        return env[self.name]


def collect_symbols(f):
    sig = inspect.signature(f)
    if len(sig.parameters) < 2:
        raise ValueError(f"method {f.__name__} should have at least 2 "
                         "parameters, 'self' and the collected symbols")
    param_list = list(sig.parameters)
    collected_symbols_arg_name = param_list[-1]
    wrapped_arg_names = list(sig.parameters)[1:-1]
    nargs_wrapped = len(wrapped_arg_names)
    
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        # Validate arguments
        nargs_in = len(args) + len(kwargs)
        if collected_symbols_arg_name in kwargs:
            msg = (f"{f.__name__}() argument '{collected_symbols_arg_name}' "
                   "is filled out by the decorator collect_symbols")
            raise TypeError(msg)
        if nargs_in != nargs_wrapped:
            raise TypeError(f"{f.__name__}() takes {nargs_wrapped} but "
                            f"{nargs_in} were given")

        # Assert that the needed model variables were specified
        model_vars = self.variables
        assert all(arg_name in model_vars for arg_name in wrapped_arg_names)
        
        # Create substitution dictionary
        subs = {}
        if 'self' in model_vars:
            subs.update(model_vars['self'].subs_dict(self))
        for name, value in zip(wrapped_arg_names, args):
            subs.update(model_vars[name].subs_dict(value))
        for name, value in kwargs.items():
            var = model_vars.get(name, None)
            if var is None:
                raise TypeError(f"{f.__name__}() got an unexpected keyword "
                                f"argument '{name}'")
            subs.update(model_vars[name].subs_dict(value))
        
        # Create collected symbols AttrDict
        symbols = attrdict.AttrDict()
        for var, sub in subs.items():
            name = getattr(var, 'name', None)
            if name is not None:
                symbols[name.split('.')[-1]] = sub
        kwargs[collected_symbols_arg_name] = symbols
        return f(self, *args, **kwargs)
    wrapper.__signature__ = utils.make_signature(wrapped_arg_names, member=True)
    return wrapper
