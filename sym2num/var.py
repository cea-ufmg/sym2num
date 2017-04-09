"""Symbolic variables for code generation."""


import numpy as np
import sympy


class Variable(sympy.Symbol):
    """Represents a code generation variable."""
        
    def validate(self):
        """Returns the assertion code to validate the variable."""
        validations = (self.validate_type(), self.validate_assumptions())
        
        return str.join("\n", (v for v in validations if v is not None))

    def validate_type(self):        
        """Returns the assertion code to validate the variable type."""
        return
    
    def validate_assumptions(self):
        """Returns the assertion code to validate the variable assumptions."""
        return
    
