"""
Monomial expression (term)
"""


class Monomial(object):
    def __init__(self, _vars, c=1, a=None):
        if isinstance(_vars, str):
            _vars = [_vars]
        N = len(_vars)
        if a is None:
            a = [1]*N
        assert N == len(a), 'N=%s but len(a)=%s' % (N, len(a))

        self.vars = _vars
        self.c = c
        self.a = a

    def __repr__(self):
        c = [str(self.c)] if self.c != 1 else []
        t = ['%s^%s' % (v, a) if a != 1 else v
             for v, a in zip(self.vars, self.a)]
        return '*'.join(c + t)
    
    def latex(self):
        c = [str(self.c)] if self.c != 1 else []
        t = ['%s^%s' % (v, a) if a != 1 else v
             for v, a in zip(self.vars, self.a)]
        return '$%s$' % ''.join(c + t)

    def __eq__(self, x):
        """Equality test

        Args: 
            x (Monomial): Monomial to compare with

        Returns:
            bool -- True if self == x
        
        Notes:
            Currently returns True only if variables have same order
        """
        return (isinstance(x, Monomial) and 
                self.c == x.c and 
                self.vars == x.vars and 
                self.a == x.a)

    def __ne__(self, x):
        return not self.__eq__(x)

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
        c = self.c/float(m.c)
        a = list(self.a)
        _vars = list(self.vars)
        for i, v in enumerate(m.vars):
            if v in _vars:
                a[i] = a[i] - m.a[i]
            else:
                _vars.append(v)
                a.append( -m.a[i] )
        return Monomial(_vars, c, a)

