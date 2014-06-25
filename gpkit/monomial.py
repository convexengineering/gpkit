"""
Monomial expression (term)
"""

# TODO(ned): make interpreter for LATEX-like strings

class Monomial(object):
    def __init__(self, _vars, c=1, a=None):
        if isinstance(_vars, str):
            _vars = [_vars]
        N = len(_vars)
        if a is None:
            a = [1]*N
        assert N == len(a), 'N=%s but len(a)=%s' % (N, len(a))

        self.c = float(c)
        self.exps = dict(zip(_vars, a))
        self.vars = set(_vars) # to sort and remove duplicates

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
        c = self.c/float(m.c)
        a = [self.exps.get(var, 0) - m.exps.get(var, 0)
             for var in _vars]
        return Monomial(_vars, c, a)

    def __mul__(self, m):
        """Multiplication by another monomial
        
        Args:
            m (Monomial): monomial to multiply by

        Returns:
            Monomial
        """
        return self.__div__(m**-1)


    def __sub__(self, m):
        """Subtraction of another monomial from this one
        
        Args:
            m (Monomial): monomial to subtract

        Returns:
            Monomial
        """
        # TODO(ned): make this actually work
        return Posynomial([self, -1*m])