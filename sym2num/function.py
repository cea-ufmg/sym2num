"""Sympy numeric function generation."""


import functools
import warnings

import jinja2
import numpy as np
import sympy

from . import utils, printing, var


class Arguments(var.Dict):
    """Represents symbolic array function arguments."""
    pass


def print_function(name, output, arguments, **options):
    return FunctionPrinter(name, output, arguments, **options).print_def()


def compile_function(name, output, arguments, **options):
    return FunctionPrinter(name, output, arguments, **options).callable()


class Arguments(var.Dict):
    pass


function_template_src = '''\
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
    {{ arg.print_prepare_validate(printer, used_symbols) | indent}}
    {%- endfor %}
    {% if cse_subs %}# Calculate the common subexpressions
    {% endif -%}
    {% for cse_symbol, cse_expr in cse_subs -%}
    {{cse_symbol}} = {{printer.doprint(cse_expr)}}
    {% endfor -%}
    {% if broadcast_elements -%}
    # Broadcast the input arguments
    _broadcast = {{np}}.broadcast({{broadcast_elements | join(', ')}})
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
        return jinja2.Template(function_template_src)
    
    def __init__(self, name, output, arguments, **options):
        self.name = name
        """Generated function name."""
        
        output = np.asarray(output, object)
        self.output = output
        """Symbolic array expression of the function's output."""
        
        self.arguments = arguments
        """Ordered dict of sym2num Variable with the function arguments."""
        
        self.options = options
        """Symbolic code generation options."""
        
        argument_ids = [a.identifiers for a in arguments.values()]
        all_argument_ids = utils.union(argument_ids)
        if sum(map(len, argument_ids)) > len(all_argument_ids):
            raise ValueError("duplicate symbols found in input argument list")
        
        output_ids = {s.name for e in output.flat for s in e.free_symbols}
        orphan_symbol_ids = output_ids - all_argument_ids
        if orphan_symbol_ids:
            msg = "symbols `{}` of the output are not in the input"
            raise ValueError(msg.format(', '.join(orphan_symbol_ids)))
        
        output_callables = {
            c.name for e in output.flat for c in e.atoms(var.CallableBase)
        }
        orphan_callables = output_callables - all_argument_ids
        if orphan_callables:
            msg = "custom callables `{}` of the output are not in the input"
            raise ValueError(msg.format(', '.join(orphan_callables)))
    
    @property
    def argument_names(self):
        """List of names of the generated function arguments."""
        return [arg.name for arg in self.arguments]
    
    @property
    def broadcast_elements(self):
        """List of argument elements broadcasted to generate the output"""
        return sum((arg.broadcast_elements for arg in self.arguments), [])
    
    def output_code(self, printer):
        """Iterator of the ndenumeration of the output code."""
        for ind in np.ndindex(*self.output.shape):
            expr = self.output[ind]
            if expr != 0:
                yield ind, printer.doprint(expr)
    
    def print_def(self):
        """Print the function definition code."""
        printer = printing.Printer()
        output_code = list(self.output_code(printer))
        broadcast_elements = self.broadcast_elements
        used_symbols = self.output.free_symbols.union(broadcast_elements)
        context = dict(
            f=self, 
            printer=printer, 
            np=printer.numpy_alias,
            output_code=output_code,
            used_symbols=used_symbols,
            broadcast_elements=broadcast_elements,
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
        self.__call__ = utils.wrap_with_signature(arg_name_list)(self.__call__)
    
    def __call__(self, *args):
        # Append self if we are a bound method
        if self.arguments[0].name == 'self':
            args = self.arguments[0], *args
        
        assert len(args) == len(self.arguments)
        
        subs = {}
        for var, value in zip(self.arguments, args):
            subs.update(var.subs_dict(value))
        
        # double substitution is needed when the same symbol appears in the
        # function definition and call arguments
        temp_subs = {s: sympy.Symbol('_temp_subs_' + s.name) for s in subs}
        final_subs = {temp_subs[s]: subs[s] for s in temp_subs}
        return self.output.subs(temp_subs).subs(final_subs)


def isstatic(arguments):
    """Return whether an argument list corresponds to a static method."""
    if len(arguments) == 0:
        return True
    elif not isinstance(arguments[0], var.SymbolObject):
        return True
    else:
        return 'cls' != arguments[0].name != 'self'


def isclassmethod(arguments):
    """Return whether an argument list corresponds to a classmethod."""
    return (len(arguments) > 0 
            and isinstance(arguments[0], var.SymbolObject)
            and arguments[0].name == 'cls')
