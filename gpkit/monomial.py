"""
Monomial expression (term)
"""

# NOTE: there's a circular import at the bottom of this file.
#   since the monomial class only calls the Posynomial class
#   when there's addition or subtraction, it might be alright?

def monify(s):
    """
    Turns a whitespace separated string into singlet monomials.
    """
    return [Monomial(x) for x in s.split()]


class Monomial(object):

    def __init__(self, _vars, c=1, a=None):
        self.c = float(c)
        self.vars = set(_vars)

        if isinstance(_vars, dict):
            # check if we already have it as a dictionary
            self.exps = _vars
        else:
            # ensure _vars is a list of variable names
            if isinstance(_vars, str):  _vars = [_vars]
            N = len(_vars)
            # if we don't have an exponent list, use the default
            if a is None:  a = [1]*N
            # if we do have one, check that it's the right length
            assert N == len(a), 'N=%s but len(a)=%s' % (N, len(a))
            # zip 'em up!
            self.exps = dict(zip(_vars, a))

    def _str_tokens(self, joiner='*'):
        t = []
        for var in self.vars:
            exp = self.exps[var]
            if exp != 0:
                t.append('%s^%g' %
                         (var, exp) if exp != 1 else var)
        c = ["%g" % self.c] if self.c != 1 or not t else []
        return joiner.join(c + t)

    def __repr__(self):
        return self._str_tokens()
    
    def latex(self, bracket='$'):
        latexstr = self._str_tokens('') # could put a space in here?
        return bracket + latexstr + bracket

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, m):
        """Equality test

        Args: 
            m (Monomial): Monomial to compare with

        Returns:
            bool -- True if self == m
        
        Notes:
            Currently returns True only if variables have same order
        """
        return (isinstance(m, Monomial) and str(self) == str(m))

    def __ne__(self, m):
        return not self.__eq__(m)

    def __pow__(self, x):
        """Put monomial to a numeric power
        
        Args:
            x (float or int): exponent to put monomial to

        Returns:
            Monomial
        """
        # assume x is a number
        a = []
        for var in self.vars:
            exp = self.exps[var]
            a.append(exp*x)
        c = self.c**x
        return Monomial(self.vars, c, a)

    def __div__(self, m):
        """Division by another monomial
        
        Args:
            m (Monomial): monomial to divide by

        Returns:
            Monomial
        """
        if not isinstance(m, Monomial):
            # assume m is numeric scalar
            m = Monomial([], c=m)
        _vars = self.vars.union(m.vars)
        c = self.c/m.c
        a = [self.exps.get(var, 0) - m.exps.get(var, 0)
             for var in _vars]
        return Monomial(_vars, c, a)

    def __rdiv__(self, m):
        # m/self
        return m * self**-1

    def __mul__(self, m):
        """Multiplication by another monomial
        
        Args:
            m (Monomial): monomial to multiply by

        Returns:
            Monomial
        """
        return self/(m**-1)

    def __rmul__(self, m):
        return self*m


    def __sub__(self, m):
        return Posynomial([self, -1*m])

    def __rsub__(self, m):
        return Posynomial([m, -1*self])

    def __add__(self, m):
        return Posynomial([self, m])

    def __radd__(self, m):
        return Posynomial([m, self])

    def __neg__(self):
        c = -self.c
        return Monomial(self.exps, c)

from posynomial import Posynomial