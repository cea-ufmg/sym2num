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


class Base:
    def __init__(self):
        self.variables = var.SymbolObject(self={})
        """Model variables dictionary."""
        
        self.init_variables()
        self.init_derivatives()
        
    def init_variables(self):
        """Initialize model variables."""
        pass
    
    def init_derivatives(self):
        """Initialize model derivatives."""
        pass

    def add_derivative(self, name, fname, wrt, flatten_wrt=False):
        pass
    
    @contextlib.contextmanager
    def use_default_members(self):
        """Context manager that sets default attributes temporarily."""
        members = {k: getattr(self, k, None) for k in self.variables['self']}
        try:
            for key, val in self.variables['self'].items():
                setattr(self, key, val)
            yield
        finally:
            for key, val in members.items():
                setattr(self, key, val)
    
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
    
    def function_codegen_arguments(self, f):
        
        if isinstance(f, function.SymbolicSubsFunction):
            return f.arguments
        
        param_names = inspect.signature(f).parameters.keys()
        return function.Arguments((n,self.variables[n]) for n in param_names)
    
    def default_function_output(self, fname):
        """Function output for the default arguments."""
        f = self._get_func(fname)
        if isinstance(f, function.SymbolicSubsFunction):
            return f.default_output
        
        args = self.function_codegen_arguments(f)
        with self.use_default_members():
            return f(*args.values())


class OldBase:
    def __init__(self):
        self._init_derivatives()
    
    def _init_derivatives(self):
        """Initialize model derivatives."""
        init_derivatives_method = getattr(self, 'init_derivatives', None)
        if init_derivatives_method:
            init_derivatives_method()
        else:
            for spec in getattr(self, 'derivatives', []):
                self.add_derivative(*spec)
    
    @property
    def variables(self):
        return var.Variables()
    
    @property
    def symbol_index_map(self):
        return var.symbol_index_map(self.variables.values())

    @property
    def array_shape_map(self):
        return var.array_shape_map(self.variables.values())
    
    @property
    def array_element_names(self):
        return var.array_element_names(self.variables.values())
    
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
            raise ValueError(f"{fname} member not found")
        except TypeError:
            raise TypeError("function name must be a string")

        if isinstance(f, function.SymbolicSubsFunction):
            return f.output
        
        args = self.function_codegen_arguments(fname)
        if len(args) > 0 and args[0].name == 'self':
            args = args[1:]
        return f(*args)
    
    def add_derivative(self, name, fname, wrt, flatten_wrt=False):
        # Test if we have only one wrt item
        if isinstance(wrt, (str, sympy.NDimArray)):
            wrt = (wrt,)
        
        out = self.default_function_output(fname)
        for wrt_array in wrt:
            if utils.isstr(wrt_array):
                wrt_array = self.variables[wrt_array]
            if flatten_wrt:
                wrt_array = sympy.flatten(wrt_array)
            out = sympy.derive_by_array(out, wrt_array)
        
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
{% for import in m.imports -%}
import {{ import }}
{% endfor %}
class {{m.name}}({{ m.bases | join(', ') }}):
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
        self.model = model
        """The underlying symbolic model."""
        
        self.options = options
        """Model printer options."""
        
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
            return getattr(self.model, 'generated_bases', [])
    
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


class ModelArrayInitializer:
    def __init__(self, **kwargs):
        # Create all arrays
        for name, shape in self.array_shape_map.items():
            if name.startswith('self.') and not name.count('.', 5):
                setattr(self, name[5:], np.zeros(shape))
        
        # Initialize the arguments
        symbol_index_map = self.symbol_index_map
        for symbol_name, value in kwargs.items():
            with contextlib.suppress(KeyError):
                array_name, index = symbol_index_map[symbol_name]
                array_name_parts = array_name.split('.')
                if len(array_name_parts) == 2 and array_name_parts[0] == 'self':
                    getattr(self, array_name_parts[1])[index] = value

