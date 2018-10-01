"""Symbolic variables for code generation."""


import collections
import inspect
import keyword
import numbers

import jinja2
import numpy as np
import sympy
from sympy.core.function import ArgumentIndexError

from . import utils


class Variable:
    """Represents a code generation variable."""
        
    def print_prepare_validate(self, printer):
        """Returns code to validate and prepare the variable from arguments."""
        return ''
    
    @property
    def broadcast_elements(self):
        """List of elements which should be broadcast to generate the output."""
        return []
    
    def subs_dict(self, value):
        """Dictionary of substitutions to evaluate with a given value."""
        return {self: value}
    
    @property
    def identifiers(self):
        """Set of identifiers defined in this variable's code."""
        return {self.name}


class SymbolObject(Variable):
    def __init__(self, name, *args):
        self.name = name
        self.members = {}
        for v in args:
            self.members[v.name] = v
            v.name = f'{name}.{v.name}'
    
    def __getattr__(self, name):
        try: 
            return self.members[name]
        except KeyError:
            raise AttributeError(f'SymbolObject has no attribute {name}')
    
    def print_prepare_validate(self, printer):
        """Returns code to validate and prepare the variable from arguments."""
        members = self.members.values()
        lines = (v.print_prepare_validate(printer) for v in members)
        return str.join('\n', (line for line in lines if line))
    
    @property
    def broadcast_elements(self):
        """List of elements which should be broadcast to generate the output."""
        out = []
        for v in self.members.values():
            out.extend(v.broadcast_elements)
        return out
    
    def subs_dict(self, value):
        """Dictionary of substitutions to evaluate with a given value."""
        out = {self: value}
        for name, v in self.members.items():
            out.update(v.subs_dict(getattr(value, name)))
        return out
    
    @property
    def identifiers(self):
        """Set of identifiers defined in this variable's code."""
        out = {self.name}
        for v in self.members.values():
            out.update(v.identifiers)
        return out



class SymbolArray(Variable, sympy.Array):
    """Represents array of symbols for code generation."""
    
    def __new__(cls, name, array_like=None, dtype='float64'):
        if array_like is None:
            array_like = name
        
        elements, shape = elements_and_shape(array_like)
        if all(utils.isstr(e) for e in elements):
            elements = [sympy.Symbol(n) for n in elements]
        
        if len(set(elements)) != len(elements):
            raise ValueError("elements of SymbolArray must be unique")
        
        obj = super().__new__(cls, elements, shape)
        return obj
    
    def __init__(self, name, array_like=None, dtype='float64'):
        if not isinstance(name, str):
            raise TypeError("expected str, but got {!r}".format(type(name)))
        if not utils.isidentifier(name):
            raise ValueError(
                "'{}' is not a valid python identifier".format(name)
            )
        self.name = name
        """Variable name"""
        
        self.dtype = dtype
        """Generated array dtype."""

        symbol_names = set(symbol.name for symbol in self)
        if len(self) > len(symbol_names):
            raise ValueError("symbol names in array must be unique")
        
        if self.rank() > 0 and self.name in symbol_names:
            raise ValueError("positive-rank array name and symbols must differ")
        
    def __len__(self):
        """Overrides `sympy.Array.__len__` which fails for rank-0 Arrays"""
        if self.shape == ():
            return 1
        else:
            return super().__len__()

    def __getitem__(self, index):
        return sympy.Array(self, self.shape)[index]

    def ndenumerate(self):
        for ind in np.ndindex(*self.shape):
            yield ind, self[ind]
    
    @property
    def broadcast_elements(self) -> list:
        """List of elements which should be broadcast to generate the output."""
        return [] if len(self) == 0 else [self[(0,) * self.rank()]]
    
    def subs_dict(self, value):
        value_array = sympy.Array(value)
        if self.shape != value_array.shape:
            msg = "Invalid shape for argument, expected {} and got {}"
            raise ValueError(msg.format(self.shape, value_array.shape))
        
        subs = {self: value}
        for i in np.ndindex(*self.shape):
            subs[self[i]] = value_array[i]
        return subs
    
    @utils.cached_class_property
    def prepare_validate_template(cls):
        return jinja2.Template(inspect.cleandoc("""
        {% if v.ensure_array -%}
        {{v.name}} = {{np}}.asarray({{v.name}}, dtype={{np}}.{{v.dtype}})
        {% endif -%}
        {% if v.rank() -%}
        if {{v.name}}.shape[-{{v.rank()}}:] != {{v.shape}}:
        {%- set expected %}(...,{{v.shape |join(",")}}){% endset %}
            msg = "invalid shape for {{v.name}}, expected {{expected}}, got {}"
            raise ValueError(msg.format({{v.name}}.shape))
        {% endif -%}
        {% if v.rank() != 0 or v.name != v[()].name -%}
        # unpack `{{v.name}}` array elements
        {% for ind, symb in v.ndenumerate() -%}
        {{symb}} = {{v.name}}[..., {{ ind | join(', ')}}]
        {% endfor %}
        {%- endif %}
        """))
    
    def print_prepare_validate(self, printer):
        """Construct variable from an array_like and check dimensions."""
        context = dict(v=self, np=printer.numpy_alias, printer=printer)
        return self.prepare_validate_template.render(context)

    @property
    def identifiers(self):
        """Set of identifiers defined in this variable's code."""
        return {self.name} | {elem.name for elem in self}

    @property
    def ensure_array(self):
        """Whether to use `np.asarray` to ensure the argument is an ndarray."""
        return '.' not in self.name


class CallableBase(sympy.Function):
    """Base class for code-generation callables like in `scipy.interpolate`."""
    
    @classmethod
    def print_prepare_validate(cls, printer):
        """Returns code to validate and prepare the variable from arguments."""
        return ''
    
    @utils.classproperty
    def broadcast_elements(cls):
        """List of elements which should be broadcast to generate the output."""
        return []
    
    @classmethod
    def subs_dict(cls, value):
        """Dictionary of substitutions to evaluate with a given value."""
        name = getattr(cls, 'name', None) or cls.__name__
        return {cls: value}
    
    @utils.classproperty
    def identifiers(cls):
        """Set of identifiers defined in this variable's code."""
        return {cls.name}


class UnivariateCallableBase(CallableBase):
    """Base for univariate callables like Pchip, PPoly, Akima1d, Spline, etc."""
    
    nargs = (1, 2)
    """Number of function arguments."""
    
    @property
    def dx(self):
        if len(self.args) == 1:
            return 0
        assert isinstance(self.args[1], (numbers.Integral, sympy.Integer))
        return self.args[1]
    
    def fdiff(self, argindex=1):
        if argindex == 2:
            raise ValueError("Only derivatives wrt first argument allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        dx = self.dx
        return self.__class__(self.args[0], dx + 1)


class BivariateCallableBase(CallableBase):
    """Base for bivariate callables like scipy's BivariateSpline."""
    
    nargs = (2, 4)
    """Number of function arguments."""
    
    @property
    def dx(self):
        if len(self.args) == 2:
            return 0
        assert isinstance(self.args[2], (numbers.Integral, sympy.Integer))
        return self.args[2]
    
    @property
    def dy(self):
        if len(self.args) == 2:
            return 0
        assert isinstance(self.args[3], (numbers.Integral, sympy.Integer))
        return self.args[3]
    
    def fdiff(self, argindex=1):
        if argindex > 2:
            raise ValueError("Only derivatives wrt x and y allowed")
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        dx = self.dx + 1 if argindex == 1 else self.dx
        dy = self.dy + 1 if argindex == 2 else self.dy
        return self.__class__(*self.args[:2], dx, dy)


def UnivariateCallable(name):
    metaclass = type(UnivariateCallableBase)
    return metaclass(name, (UnivariateCallableBase,), {'name': name})


def BivariateCallable(name):
    metaclass = type(BivariateCallableBase)
    return metaclass(name, (BivariateCallableBase,), {'name': name})


def elements_and_shape(array_like):
    """Return flat list of elements and shape from array-like nested iterable.
    
    Based on `sympy.tensor.array.ndim_array.NDimArray._scan_iterable_shape`.
    """
    #Detect if we are at a scalar element
    if (utils.isstr(array_like) or not utils.isiterable(array_like)):
        return [array_like], ()

    #Detect empty iterable
    array_like = list(array_like)
    if len(array_like) == 0:
        return [], (0,)
    
    #We have an iterable, apply self to its elements
    subelements, subshapes = zip(*[elements_and_shape(e) for e in array_like])
    
    #Check if all subelements have the same shape
    if len(set(subshapes)) != 1:
        raise ValueError("could not determine shape unambiguously")
    
    #Create outputs
    elements = []
    for subelement in subelements:
        elements.extend(subelement)
    shape = (len(subelements),) + subshapes[0]
    return elements, shape


def make_dict(var_list):
    """Make a dictionary from a variables list."""
    return collections.OrderedDict((var.name, var) for var in var_list)


def symbol_index_map(iterable):
    """Return mapping of array element names to variable name and index."""
    m = {}
    for var in iterable:
        if isinstance(var, SymbolObject):
            m.update(symbol_index_map(var.members.values()))
        elif isinstance(var, SymbolArray):
            for ind, symbol in var.ndenumerate():
                if symbol.name not in m:
                    m[symbol.name] = (var.name, ind)
    return m


def array_shape_map(iterable):
    """Return mapping of variable names to array shapes."""
    m = {}
    for var in iterable:
        if isinstance(var, SymbolObject):
            m.update(array_shape_map(var.members.values()))
        elif isinstance(var, SymbolArray):
            m[var.name] = var.shape
    return m


def array_element_names(iterable):
    """Return mapping of array indices to element names."""
    out = {}
    for var in iterable:
        if isinstance(var, SymbolObject):
            out[var.name] = array_element_names(var.members.values())
        elif isinstance(var, SymbolArray):
            out[var.name] = {ind: elem.name for ind,elem in var.ndenumerate()}
    return out
