"""Sympy numeric function generation."""


import collections
import collections.abc as colabc
import functools
import itertools
import keyword
import re
import warnings

import jinja2
import numpy as np
import sympy

from . import utils, printing, var


def print_function(name, output, arguments, **options):
    return FunctionPrinter(name, output, arguments, **options).code()


def compile_function(name, output, arguments, **options):
    return FunctionPrinter(name, output, arguments, **options).callable()


numpy_function_template_src = '''\
def {{f.name}}({{f.argument_names | join(', ')}}):
    """Generated function `{{f.name}}` from sympy Array expression."""
    # Function imports
    import numpy as {{np}}
    {% for mod in printer.direct_imports if mod != 'numpy' -%}
    import {{mod}}
    {% endfor -%}
    {% for mod, alias in printer.aliased_imports if mod != 'numpy' -%}
    import {{mod}} as {{alias}}
    {% endfor %}
    # Prepare and validate arguments
    {%- for arg in f.arguments %}
    {{ arg.print_prepare_validate(printer) | indent}}
    {%- endfor %}
    {% if cse_subs %}# Calculate the common subexpressions
    {% endif -%}
    {% for cse_symbol, cse_expr in cse_subs -%}
    {{cse_symbol}} = {{printer.doprint(cse_expr)}}
    {% endfor -%}
    {% if f.broadcast_elements -%}
    # Broadcast the input arguments
    _broadcast = {{np}}.broadcast({{f.broadcast_elements | join(', ')}})
    _out = {{np}}.zeros(_broadcast.shape + {{f.output.shape}})
    {% else -%}
    _out = {{np}}.zeros({{f.output.shape}})
    {% endif %}
    # Assign the nonzero elements of the output
    {% for ind, expr in output_code if expr != 0 -%}
    _out[..., {{ind | join(', ')}}] = {{expr}}
    {% endfor -%}
    return _out
'''

class FunctionPrinter:
    """Generates numpy code for symbolic array functions."""

    @utils.cached_class_property
    def template(cls):
        return jinja2.Template(numpy_function_template_src)
    
    def __init__(self, name, output, arguments, **options):
        self.name = name
        """Generated function name."""

        if not isinstance(output, sympy.NDimArray):
            warnings.warn("sympy.NDimArray instance expected as output")
        self.output = output
        """Symbolic expression of the function's output."""
        
        self.arguments = arguments
        """List of sym2num Variables with the function arguments."""
        
        self.options = options
        """Symbolic code generation options."""
        
        argument_ids = [a.identifiers for a in arguments]
        all_argument_ids = functools.reduce(set.union, argument_ids, set())
        if sum(map(len, argument_ids)) > len(all_argument_ids):
            raise ValueError("duplicate symbols found in input argument list")
        
        output_ids = {s.name for s in output.free_symbols}
        orphan_symbol_ids = output_ids - all_argument_ids
        if orphan_symbol_ids:
            msg = "symbols `{}` of the output are not in the input"
            raise ValueError(msg.format(', '.join(orphan_symbol_ids)))
    
    @property
    def argument_names(self):
        """List of names of the generated function arguments."""
        return [arg.name for arg in self.arguments]
    
    @property
    def broadcast_elements(self):
        """List of argument elements broadcasted to generate the output"""
        return sum((arg.broadcast_elements for arg in self.arguments), [])
    
    def output_code(self, printer):
        """Iterator of the ndenumeration of the output."""
        for ind in np.ndindex(*self.output.shape):
            expr = self.output[ind]
            if expr != 0:
                yield ind, printer.doprint(expr)
    
    def print_def(self):
        """Print the function definition code."""
        printer = printing.Printer()
        output_code = list(self.output_code(printer))
        context = dict(
            f=self, 
            printer=printer, 
            np=printer.numpy_alias,
            output_code=output_code
        )
        return self.template.render(context)

    def callable(self):
        env = {}
        exec(compile(self.print_def(), '<string>', 'exec'), env)
        return utils.wrap_with_signature(self.argument_names)(env[self.name])


class SymbolicSubsFunction:
    def __init__(self, arguments, output):
        self.arguments = tuple(arguments)
        self.output = output
        
        arg_name_list = [a.name for a in arguments]
        self.__call__ = wrap_with_signature(arg_name_list)(self.__call__)
    
    def __call__(self, *args):
        assert len(args) == len(self.arguments)
        subs = {}
        for var, value in zip(self.arguments, args):
            subs.update(var.subs_dict(value))
        
        # double substitution is needed when the same symbol appears in the
        # function definition and call arguments
        temp_subs = {s: sympy.Symbol('_temp_subs_' + s.name) for s in subs}
        final_subs = {temp_subs[s]: subs[s] for s in temp_subs}
        return self.output.subs(temp_subs).subs(final_subs)


function_template = '''\
def {{symfun.name}}({{symfun.args | join(', ')}}):
    """Generated function `{{symfun.name}}` from sympy ndarray expression."""
    # Convert arguments to ndarrays and create aliases to prevent name conflicts
    {% for argname in symfun.args -%}
    _{{argname}}_asarray = {{printer.numpy}}.asarray({{argname}})
    {% endfor %}
    # Check argument lengths
    {% for argname, arg in symfun.args.items() -%}
    {% if arg.shape -%}
    if _{{argname}}_asarray.shape[-{{arg.ndim}}:] != {{arg.shape}}:
        raise ValueError("Invalid dimensions for argument `{{argname}}`.")
    {% endif -%}
    {% endfor %}
    # Unpack the elements of each argument
    {% for argname, arg in symfun.args.items() -%}
    {% for index, element in np.ndenumerate(arg) -%}
    {% if element in free_symbols or element in broadcast -%}
    {%- set fullindex = ('...',) + index -%}
    {{element}} = _{{argname}}_asarray[{{fullindex | join(', ')}}]
    {% endif -%}
    {% endfor -%}
    {% endfor %}
    {%- if cse_subs %}
    # Calculate the common subexpressions
    {% for cse_symbol, cse_expr in cse_subs -%}
    {{cse_symbol}} = {{printer.doprint(cse_expr)}}
    {% endfor -%}
    {%- endif %}
    # Broadcast the input arguments
    {% if broadcast_len > 1 -%}
    _broadcast = {{printer.numpy}}.broadcast({{broadcast | join(', ')}})
    _base_shape = _broadcast.shape
    {% elif broadcast_len == 1 -%}
    _base_shape = {{broadcast[0]}}.shape
    {% else -%}
    _base_shape = ()
    {% endif -%}
    _out = {{printer.numpy}}.zeros(_base_shape + {{out.shape}})

    # Assign the nonzero elements of the output
    {% for index, expr in np.ndenumerate(out) if expr != 0 -%}
    {%- set fullindex = ('...',) + index -%}
    _out[{{fullindex | join(', ')}}] = {{printer.doprint(expr)}}
    {% endfor -%}
    return _out'''


def symbol_array(obj):
    # Convert to python object array
    arr = np.array(obj, dtype=object)
    
    # Check array contents
    for index, elem in np.ndenumerate(arr):
        if not isinstance(elem, sympy.Symbol):
            msg = "Array elements should be `sympy.Symbol`, got `{}.{}`."
            msg = msg.format(type(elem).__module__, type(elem).__name__)
            raise TypeError(msg)
        if keyword.iskeyword(elem.name):
                msg = "Element index `{}` is python keyword `{}`."
                raise ValueError(msg.format(index, elem.name))
        if not elem.name.isidentifier():
            msg = "Element `{}` at index `{}` is not a valid python identifier."
            raise ValueError(msg.format(index, elem.name))
        if elem.name.startswith('_'):
            warnings.warn("Symbols starting with '_' may conflict " +
                          "with internal generated variables.")
    
    return arr


class SymbolicFunction:
    def __init__(self, f, args={}, name=None):
        # Process and save the input arguments
        arg_items = args.items() if isinstance(args, colabc.Mapping) else args
        arg_items = [(name, symbol_array(arg)) for name, arg in arg_items]
        self.args = collections.OrderedDict(arg_items)
        self.out = np.asarray(f(*self.args.values()) if callable(f) else f)
        self.name = name or (f.__name__ if callable(f) else None)

        # Get the free symbols of the output        
        self.free_symbols = set()
        for expr in self.out.flat:
            if isinstance(expr, sympy.Expr):
                self.free_symbols.update(expr.free_symbols)
        
        # Check if a valid name was found
        if name is None and not callable(f):
            raise TypeError("A name must be provided if f is not callable.")
        if not self.name:
            raise ValueError("Invalid function name for symbolic function.")
        
        # Check for duplicate symbols
        arg_elements = utils.flat_cat(**self.args)
        unique_arg_elements = set(arg_elements)
        if len(unique_arg_elements) != arg_elements.size:
            raise ValueError('Duplicate symbols found in function arguments.')
    
    def print_def(self, printer):
        # Check for conflicts between function and printer symbols
        for elem in utils.flat_cat(**self.args):
            if elem.name in printer.modules:
                msg = "Function argument {} conflicts with printer module."
                raise ValueError(msg.format(elem.name))

        # Perform common subexpression elimination
        cse_symbols = sympy.numbered_symbols('_cse')
        cse_in = [sympy.sympify(expr) for expr in self.out.flat]
        if cse_in:
            cse_subs, cse_exprs = sympy.cse(cse_in, cse_symbols)
            out = np.zeros_like(self.out)
            out.flat = cse_exprs
        else:
            cse_subs = []
            out = self.out
        
        # Create the template substitution tags
        broadcast = [a.flat[0] for a in self.args.values() if a.size > 0]
        tags = {
            'symfun': self,
            'printer': printer,
            'np': np,
            'free_symbols': self.free_symbols,
            'cse_subs': cse_subs,
            'broadcast': broadcast,
            'broadcast_len': len(broadcast),
            'out': out
        }
        
        # Render and return
        return jinja2.Template(function_template).render(tags)
    
    def print_module(self, printer):
        imports = '\n'.join(printer.imports)
        def_code = self.print_def(printer)
        return '{}\n\n{}'.format(imports, def_code)
    
    def diff(self, wrt, name):
        diff = utils.ndexpr_diff(self.out, wrt)
        return type(self)(diff, args=self.args, name=name)

