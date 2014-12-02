'''Sympy printer for numeric code generation.'''


import sympy
from sympy import printing


class NumpyPrinter(printing.str.StrPrinter):
    '''Sympy printer for generating python code using numpy.'''
    
    printmethod = '_numpyrepr'
    
    def _print_atan2(self, expr):
        return '_np.arctan2(%s)' % (self.stringify(expr.args, ', '),)
    
    def _print_Function(self, expr):
        return '_np.' + super()._print_Function(expr)
    
    def _print_Pi(self, expr):
        return '_np.pi'
    
    def _print_Pow(self, expr):
        if isinstance(expr.args[1], numbers.Half):
            return '_np.sqrt(%s)' % (self._print(expr.args[0]),)
        else:
            return super()._print_Pow(expr)


class ScipyPrinter(NumpyPrinter):
    '''Sympy printer for generating python code using scipy.'''
    
    def _print_erf(self, expr):
        return '_scipy_special.erf(%s)' % (self._print(expr.args[0]),)
    
    def _print_loggamma(self, expr):
        return '_scipy_special.gammaln(%s)' % (self._print(expr.args[0]),)
