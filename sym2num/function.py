"""Sympy numeric function generation."""


import functools
import warnings

import jinja2
import numpy as np
import sympy

from . import utils, printing, var


class Arguments(var.SymbolObject):
    """Represents symbolic array function arguments."""
    pass


def print_function(name, output, arguments, **options):
    return FunctionPrinter(name, output, arguments, **options).print_def()


def compile_function(name, output, arguments, **options):
    return FunctionPrinter(name, output, arguments, **options).callable()


class Arguments(var.SymbolObject):
    pass


function_template_src = '''\
def {{f.name}}({{f.argument_names | join(', ')}}):
    """Generated function `{{f.name}}` from sympy array expression."""
    # Function imports
    import numpy as {{np}}
    {% for mod in printer.direct_imports if mod != 'numpy' -%}
    import {{mod}}
    {% endfor -%}
    {% for mod, alias in printer.aliased_imports if mod != 'numpy' -%}
    import {{mod}} as {{alias}}
    {% endfor %}
    # Process and convert all arguments to ndarray
    {%- for argname, arg in f.array_arguments() %}
    _{{argname}}_asarray = {{np}}.asarray({{argname}}, dtype={{arg.gen_dtype}})
    {%- endfor %}
    {%- for argname, fname in f.callable_arguments() 
            if fname in f.referenced_callables %}
    {{fname}} = {{argname}}
    {%- endfor %}
    {%- for argname, arg in f.array_arguments() %}
    {% if arg.ndim -%}
    # Check shape of {{argname}}
    if _{{argname}}_asarray.shape[-{{arg.ndim}}:] != {{arg.shape}}:
        {%- set expected %}(...,{{arg.shape |join(",")}}){% endset %}
        shape = _{{argname}}_asarray.shape
        msg = f'wrong shape for {{argname}}, expected {{expected}}, got {shape}'
        raise ValueError(msg)
    {% endif %}    
    # Unpack {{argname}}
    {% for ind, symbol in arg.ndenumerate() if symbol in used_symbols -%}
    {{symbol}} = _{{argname}}_asarray[..., {{ind | join(", ")}}]
    {% endfor -%}
    {%- endfor %}
    {%- for argname, arg in f.object_arguments() %}
    # Unpack {{argname}}
    {% for attr, ind, symbol in arg.ndenumerate() if symbol in used_symbols -%}
    {{symbol}} = {{argname}}.{{attr}}[..., {{ind | join(", ")}}]
    {% endfor -%}
    {% for attr, fname in arg.callables() if fname in f.referenced_callables -%}
    {{fname}} = {{argname}}.{{attr}}
    {% endfor -%}
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
        
        output_ids = {symb.name for symb in self.output_symbols}
        orphan_symbol_ids = output_ids - all_argument_ids
        if orphan_symbol_ids:
            msg = "symbols `{}` of the output are not in the input"
            raise ValueError(msg.format(', '.join(orphan_symbol_ids)))
        
        orphan_callables = self.referenced_callables - all_argument_ids
        if orphan_callables:
            msg = "custom callables `{}` of the output are not in the input"
            raise ValueError(msg.format(', '.join(orphan_callables)))
    
    @property
    def argument_names(self):
        """List of names of the generated function arguments."""
        return self.arguments.keys()
    
    def array_arguments(self):
        """Iterator of the SymbolArray arguments."""
        for name, arg in self.arguments.items():
            if isinstance(arg, var.SymbolArray):
                yield name, arg

    def object_arguments(self):
        """Iterator of the SymbolObject arguments."""
        for name, arg in self.arguments.items():
            if isinstance(arg, var.SymbolObject):
                yield name, arg

    def callable_arguments(self):
        """Iterator of the callable arguments."""
        for name, arg in self.arguments.items():
            if isinstance(arg, var.CallableMeta):
                yield name, arg.name
    
    @property
    def broadcast_elements(self):
        """List of argument elements broadcasted to generate the output"""
        be = set()
        for arg in self.arguments.values():
            if isinstance(arg, var.SymbolArray):
                be.add(arg.flat[0])
        return be
    
    @property
    def output_symbols(self):
        """Set of free symbols of the output."""
        return utils.union(e.free_symbols for e in self.output.flat)
    
    @property
    def referenced_callables(self):
        """Set of the name of callables referenced by the function."""
        atoms = utils.union(e.atoms(var.CallableBase) for e in self.output.flat)
        return {c.fname for c in atoms}
    
    def output_code(self, printer):
        """Iterator of the ndenumeration of the output code."""
        for ind, expr in np.ndenumerate(self.output):
            if expr != 0:
                yield ind, printer.doprint(expr)
    
    def print_def(self):
        """Print the function definition code."""
        printer = printing.Printer()
        output_code = list(self.output_code(printer))
        broadcast_elements = self.broadcast_elements
        used_symbols = self.output_symbols.union(broadcast_elements)
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
        self.arguments = arguments
        
        self.callable_replacements = {}
        """Cache of callable replacements."""

        self.default_output = np.asarray(output, object)
        """The output for the default arguments."""
        
        # Create first round of substitutions. Double substitution is needed
        # because the same symbol may appear in the function definition and
        # call arguments.
        subs = []
        for arg in arguments.values():
            for s in arg.symbols:
                subs.append((s, self.replacement(s)))
        
        self.output_template = np.empty(self.default_output.shape, dtype=object)
        for ind, expr in np.ndenumerate(self.default_output):
            self.output_template[ind] = sympy.sympify(expr).subs(subs)
    
    def replacement(self, s):
        if isinstance(s, sympy.Symbol):
            return sympy.Symbol(f'_temp_subs_{s.name}')
        elif isinstance(s, type) and issubclass(s, var.CallableBase):
            # Caching necessary as callables with the same name are not equal
            try:
                return self.callable_replacements[s]
            except KeyError:
                if issubclass(s, var.UnivariateCallableBase):
                    sub = var.UnivariateCallable(f'_temp_subs_{s.name}')
                elif issubclass(s, var.BivariateCallableBase):
                    sub = var.BivariateCallable(f'_temp_subs_{s.name}')
                self.callable_replacements[s] = sub
                return sub
        else:
            raise TypeError
    
    def __call__(self, *args):
        if len(args) != len(self.arguments):
            msg = f'got {len(args)} arguments of {len(self.arguments)} required'
            raise TypeError(msg)

        subs = {}
        for arg, value in zip(self.arguments.values(), args):
            for key, val in arg.subs_map(value).items():
                subs[self.replacement(key)] = val
        
        output = np.empty(self.output_template.shape, object)
        for ind, expr in np.ndenumerate(self.output_template):
            output[ind] = sympy.sympify(expr).subs(subs)
        
        return output
