"""Symbolic variables for code generation."""


import collections

import numpy as np
import sympy


class Variable:
    """Represents a code generation variable."""
        
    def validate(self):
        """Returns the assertion code to validate the variable."""
        validations = (self.validate_type(), self.validate_assumptions())
        
        return str.join("\n", (v for v in validations if v))
    
    def validate_type(self):
        """Returns the assertion code to validate the variable type."""
        return
    
    def validate_assumptions(self):
        """Returns the assertion code to validate the variable assumptions."""
        return
    

class SymbolArray(Variable, sympy.Array):
    """An array of symbols for code generation."""
    
    default_assumptions = dict(real=True)
    """Default assumptions for underlying symbols."""

    def __new__(cls, array_like, shape=None):
        #If only an array_like of `str` is given, make symbols out of it
        if shape is None:
            elements, shape = elements_and_shape(array_like)
            if all(isstr(e) for e in elements):
                array_like = [
                    sympy.Symbol(n, *cls.default_assumptions) for n in elements
                ]
        obj = super().__new__(cls, array_like, shape)
        return obj


def elements_and_shape(array_like):
    """Return flat list of elements and shape from array-like nested iterable.
    
    Based on `sympy.tensor.array.ndim_array.NDimArray._scan_iterable_shape`.
    """
    #Detect if we are at a scalar element
    if (isstr(array_like) or not isiterable(array_like)):
        return [array_like], ()
    
    #We have an iterable, apply self to its elements
    subelements, subshapes = zip(*[elements_and_shape(e) for e in array_like])
    
    #Check for inconsistent size of subelements
    if len(set(subshapes)) != 1:
        raise ValueError("could not determine shape unambiguously")
    
    #Create outputs
    elements = []
    for subelement in subelements:
        elements.extend(subelement)
    shape = (len(subelements),) + subshapes[0]
    return elements, shape


def isstr(obj):
    """Return whether an object is instance of `str`."""
    return isinstance(obj, str)


def isiterable(obj):
    """Return whether an object is iterable."""
    return isinstance(obj, collections.Iterable)

