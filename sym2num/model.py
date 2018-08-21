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

from . import function, utils, printing


class Base:
    def __init__(self):
        self._init_derivatives()

    def _init_derivatives(self):
        """Initialize model derivatives."""
        for spec in getattr(self, 'derivatives', []):
            self.add_derivative(*spec)
    
    def default_function_arguments(self, fname):
        """Function output for the default arguments."""
        try:
            f = getattr(self, fname)
        except AttributeError:
            raise ValueError("{} member not found".format(fname))
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
        
        args = self.default_function_arguments(fname)
        return f(*args)
    
    def add_derivative(self, name, fname, wrt_names):
        if isinstance(wrt_names, str):
            wrt_names = (wrt_names,)
        
        out = self.default_function_output(fname)
        for wrt_name in reversed(wrt_names):
            wrt = self.variables[wrt_name]
            out = sympy.derive_by_array(out, wrt)
        
        args = self.default_function_arguments(fname)
        deriv = function.SymbolicSubsFunction(args, out)
        setattr(self, name, deriv)


def print_class(name, model, **options):
    model_printer = ModelPrinter(name, model, **options)
    return model_printer.print_class()


def compile_class(name, model, **options):
    model_printer = ModelPrinter(name, model, **options)
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
    
    def __init__(self, name, model, **options):
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
            arguments = self.model.default_function_arguments(fname)
            f_specs.append((fname, output, arguments))
        
        sparse_indices = collections.OrderedDict()
        for spec in sparse:
            fname, selector = (spec, None) if utils.isstr(spec) else spec
            output = model.default_function_output(fname)
            arguments = model.default_function_arguments(fname)
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


class_template = '''\
class {{generated_name}}({{class_signature}}):
    """Generated code for symbolic model {{sym_name}}"""

    signatures = {{signatures}}
    """Model function signatures."""

    var_specs = {{specs}}
    """Specification of the model variables."""
    {% for name, value in sparse_inds.items() %}
    {{name}}_ind = {{value}}
    """Nonzero indices of `{{name}}`."""
    {% endfor -%}
    {% for function in functions %}
    @staticmethod
    {{ function | indent}}
    {% endfor -%}
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
        self._init_sparse()
    
    def _init_variables(self):
        """Initialize the model variables."""
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
        """Initialize the model functions."""
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
        """Initialize model derivatives."""
        for spec in getattr(self, 'derivatives', []):
            self.add_derivative(*spec)
    
    def _init_sparse(self):
        """Initialize the sparse functions."""
        self.sparse_inds = {}
        for spec in getattr(self, 'sparse', []):
            if isinstance(spec, str):
                fname = spec
                selector = lambda *inds: np.ones_like(inds[0], dtype=bool)
            else:
                fname, selector = spec
            f = self.functions[fname]
            inds = np.nonzero(f.out)
            inds = [ind[selector(*inds)] for ind in inds]
            fval = f.out[inds]
            fobj = function.SymbolicFunction(fval, f.args, fname + '_val')
            self.functions[fobj.name] = fobj
            self.sparse_inds[fname] = tuple(ind.tolist() for ind in inds)
    
    @property
    @abc.abstractmethod
    def var_names(self):
        """List of the model variable names."""
        raise NotImplementedError("Pure abstract method.")

    @property
    @abc.abstractmethod
    def function_names(self):
        """List of the model function names."""
        raise NotImplementedError("Pure abstract method.")
    
    @property
    def imports(self):
        meta = getattr(self, 'meta', None)
        if callable(meta):
            return ('import ' + meta.__module__,)
        else:
            return ()
    
    @property
    def generated_name(self):
        """Name of generated class."""
        return type(self).__name__
    
    @property
    def class_signature(self):
        """Generated model class signature with metaclass definition."""
        meta = getattr(self, 'meta', None)
        if meta is None:
            return ''
        elif isinstance(meta, str):
            return 'metaclass=' + meta
        else:
            return 'metaclass={}.{}'.format(meta.__module__, meta.__qualname__)
    
    def pack(self, name, d={}, **kwargs):
        d = dict(d, **kwargs)
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
    
    def print_class(self, printer):
        tags = dict(
            generated_name=self.generated_name, 
            specs=self.var_specs, 
            sym_name=type(self).__name__,
            class_signature=self.class_signature,
            sparse_inds=self.sparse_inds,
        )
        tags['signatures'] = {name: list(f.args) 
                              for name, f in self.functions.items()}
        tags['functions'] = [fsym.print_def(printer)
                             for fsym in self.functions.values()]
        return jinja2.Template(class_template).render(tags)

    def print_module(self, printer):
        imports = '\n'.join(printer.imports + self.imports)
        class_code = self.print_class(printer)
        return '{}\n\n{}'.format(imports, class_code)
    
    def add_derivative(self, name, fname, wrt_names):
        if isinstance(wrt_names, str):
            wrt_names = (wrt_names,)
        
        f = self.functions[fname]
        for wrt_name in reversed(wrt_names):
            f = f.diff(self.vars[wrt_name], name)
        
        self.functions[name] = f


def class_obj(model, printer):
    code = model.print_module(printer)
    context = {}
    exec(code, context)
    return context[model.generated_name]


class ParametrizedModel:
    def __init__(self, params={}):
        # Save a copy of the given params
        self._params = {k: np.asarray(v) for k, v in params.items()}
        
        # Add default parameters for the empty variables
        for name, spec in self.var_specs.items():
            if np.size(spec) == 0 and name not in params:
                self._params[name] = np.array([])
    
    def parametrize(self, params={}, **kwparams):
        """Parametrize a new class instance with the given + current params."""
        new_params = self._params.copy()
        new_params.update(params)
        new_params.update(kwparams)
        return type(self)(new_params)
    
    def call_args(self, f, *args, **kwargs):
        fargs = self.signatures[f.__name__]
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
    def pack(cls, name, d, fill=0):
        spec = np.array(cls.var_specs[name])
        fill = np.asarray(fill)
        ret = np.zeros(fill.shape + spec.shape)
        ret[...] = fill[(...,) + (None,) * spec.ndim]
        for index, elem_name in np.ndenumerate(spec):
            try:
                ret[(...,) + index] = d[elem_name]
            except KeyError:
                pass
        return ret


def filterout_none(d):
    """Returns a mapping without the values which are `None`."""
    items = d.items() if isinstance(d, collections.abc.Mapping) else d
    return {k: v for k, v in items if v is not None}


def make_variables_dict(variables_list_factory):
    """Make a dictionary from a variables list."""
    if callable(variables_list_factory):
        return {var.name: var for var in variables_list_factory()}
    else:
        return {var.name: var for var in variables_list_factory}


def symbols_from(names):
    name_list = [name.strip() for name in names.split(',')]
    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, *args):
            if len(args) != len(name_list):
                msg = "{} takes {} arguments ({} given)"
                raise TypeError(msg.format(f.__name__,len(name_list),len(args)))
            a = attrdict.AttrDict()
            for name, arg in zip(name_list, args):
                subs_dict = self.variables[name].subs_dict(arg)
                for symbol, value in subs_dict.items():
                    a[symbol.name] = value
            return f(self, a)
        wrapper.__signature__ = utils.make_signature(name_list, member=True)
        return wrapper
    return decorator
