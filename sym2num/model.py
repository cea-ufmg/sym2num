"""Symbolic model code generation.

Improvement ideas
-----------------
* Add compiled code to linecache so that tracebacks can be produced, like done
  in the `IPython.core.compilerop` module.

"""


import abc
import collections
import collections.abc
import contextlib
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


class Variables(var.SymbolObject):
    """Represents code generation model variables."""
    pass


class Base:
    """Code generation model base."""
    
    def __init__(self):
        self.variables = Variables(self={})
        """Model variables dictionary."""
    
        self.derivatives = {}
        """Dictionary of model derivatives, to optimize higher order diff."""
    
    def _compute_derivative(self, fname, wrt):
        assert isinstance(wrt, tuple)
        if wrt == ():
            return self.default_function_output(fname)
        
        # See if the derivative is registered
        dname = self.derivatives.get((fname,) + wrt)
        if dname is not None:
            return self.default_function_output(dname)
        
        expr = self._compute_derivative(fname, wrt[1:])
        wrt_array = self.variables[wrt[0]]
        return utils.ndexpr_diff(expr, wrt_array)
    
    def add_derivative(self, fname, wrt, dname):
        if utils.isstr(wrt):
            wrt = (wrt,)
        elif not isinstance(wrt, tuple):
            raise TypeError("argument wrt must be string or tuple")
        
        args = self.function_codegen_arguments(fname)
        expr = self._compute_derivative(fname, wrt)
        deriv = function.SymbolicSubsFunction(args, expr)
        setattr(self, dname, deriv)
        self.derivatives[(fname,) + wrt] = dname
    
    def set_default_members(self):
        for key, val in self.variables['self'].items():
            setattr(self, key, val)
    
    @contextlib.contextmanager
    def using_default_members(self):
        """Context manager that sets default attributes temporarily."""
        set_members = {}
        unset_members = []

        # Get the values of the members before the entering the context
        for k in self.variables['self'].keys():
            try:
                set_members[k] = getattr(self, k)
            except AttributeError:
                unset_members.append(k)
        
        try:
            # Set the members to their "default" values
            self.set_default_members()
            yield
        finally:
            # Restore previous values
            for key, val in set_members.items():
                setattr(self, key, val)
            for key in unset_members:
                delattr(self, key)
    
    def _get_func(self, fname):
        try:
            f = getattr(self, fname)
        except AttributeError:
            raise ValueError(f"{fname} member not found")
        except TypeError:
            raise TypeError("function name must be a string")

        if not isinstance(f, collections.abc.Callable):
            raise TypeError(f'{fname} attribute not callable')
        else:
            return f
    
    def function_codegen_arguments(self, fname):
        f = self._get_func(fname)
        if isinstance(f, function.SymbolicSubsFunction):
            param_names = f.arguments.keys()
        else:
            param_names = inspect.signature(f).parameters.keys()
        return function.Arguments((n,self.variables[n]) for n in param_names)
    
    @utils.cached_method
    def default_function_output(self, fname):
        """Function output for the default arguments."""
        f = self._get_func(fname)
        if isinstance(f, function.SymbolicSubsFunction):
            return f.default_output
        
        args = self.function_codegen_arguments(fname)
        with self.using_default_members():
            return np.asarray(f(*args.values()))

    def print_code(self, **options):
        model_printer = ModelPrinter(self, **options)
        return model_printer.print_class()

    def compile_class(self, **options):
        model_printer = ModelPrinter(self, **options)
        return model_printer.class_obj()


def print_class(model, **options):
    model_printer = ModelPrinter(model, **options)
    return model_printer.print_class()


def compile_class(model, **options):
    model_printer = ModelPrinter(model, **options)
    return model_printer.class_obj()


model_template_src = '''\
# Model imports
import numpy as {{printer.numpy_alias}}
{% for import in m.imports -%}
import {{ import }}
{% endfor %}
class {{m.name}}({{ m.bases | join(', ') }}, metaclass={{m.metaclass}}):
    """Generated code for {{m.name}} from symbolic model."""
    {% for method in m.methods %}
    {{ method | indent }}
    {% endfor %}
    {% for name, value in m.assignments.items() -%}
    {% if isndarray(value) -%}
    {{ printer.print_ndarray(value, assign_to=name) }}
    {% else -%}
    {{ name }} = {{ value }}
    {% endif -%}
    {% endfor %}
'''


class ModelPrinter:
    """Generates numpy code for symbolic models."""

    @utils.cached_class_property
    def template(cls):
        return jinja2.Template(model_template_src)
    
    def __init__(self, model, **options):
        self.model = model
        """The underlying symbolic model."""
        
        self.options = options
        """Model printer options."""
        
        try:
            functions = options['functions']
        except KeyError:
            functions = getattr(model, 'generate_functions', [])
                
        mdl_self_var = self.model.variables['self']
        base_args = function.Arguments([('self', mdl_self_var)])
        f_specs = []
        for fname in functions:
            output = self.model.default_function_output(fname)
            arguments = base_args.copy()
            arguments.update(self.model.function_codegen_arguments(fname))
            f_specs.append((fname, output, arguments))
                
        self._f_specs = f_specs
        """Function generation specifications."""
    
    @property
    def name(self):
        """Name of the generated class."""
        return (getattr(self.model, 'generated_name', None)
                or self.options.get('name', None)
                or f'Generated{type(self.model).__name__}')
    
    @property
    def assignments(self):
        """Mapping of simple assignments to be made in the class code."""
        try:
            return self.options['assignments']
        except KeyError:
            return getattr(self.model, 'generate_assignments', {})

    @property
    def imports(self):
        """List of imports to include in the generated class code."""
        try:
            return self.options['imports']
        except KeyError:
            return getattr(self.model, 'generate_imports', [])

    @property
    def bases(self):
        """List of names of base classes for the generated model class."""
        try:
            return self.options['bases']
        except KeyError:
            return getattr(self.model, 'generated_bases', ['object'])

    @property
    def metaclass(self):
        """Metaclass for the generated model class."""
        try:
            return self.options['metaclass']
        except KeyError:
            return getattr(self.model, 'generated_metaclass', 'type')
    
    @property
    def methods(self):
        for fname, output, arguments in self._f_specs:
            fdef = function.print_function(fname, output, arguments)
            yield fdef
    
    def print_class(self):
        isndarray = lambda var: isinstance(var, np.ndarray)
        context = dict(m=self, printer=printing.Printer(), isndarray=isndarray)
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
    params = list(sig.parameters.values())
    collected_symbols_arg_name = params[-1].name
    new_sig = sig.replace(parameters=params[:-1])
    nargs_wrapped = len(params) - 1
    
    @functools.wraps(f)
    def wrapper(self, *args):
        # Validate arguments
        nargs_in = len(args) + 1
        if nargs_in != nargs_wrapped:
            raise TypeError(f"{f.__name__} takes {nargs_wrapped} arguments "
                            f"but got only {nargs_in}")
        
        # Create substitution dictionary
        subs = self.variables['self'].subs_map(self)
        for param, value in zip(params[1:-1], args):
            subs.update(self.variables[param.name].subs_map(value))
        
        # Create collected symbols AttrDict
        collected_symbols = attrdict.AttrDict()
        for var, expr in subs.items():
            collected_symbols[var.name] = expr
        ret = f(self, *args, **{collected_symbols_arg_name: collected_symbols})
        
        # Ensure function return is an ndarray
        return np.asarray(ret, object)
    wrapper.__signature__ = new_sig
    return wrapper

