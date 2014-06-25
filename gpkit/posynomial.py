"""
Posynomial expression (term)
"""

# TODO(ned): make interpreter for LATEX-like strings

class Posynomial(object):
    def __init__(self, monomials):
        N = len(monomials)
        self.monomials = set(monomials)
        vars = []
        for m in self.monomials:
            vars += m.vars
        self.vars = set(vars)
    
    def __repr__(self):
        return ' + '.join([str(m) for m in self.monomials])
    
    def latex(self, bracket='$'):
        latexstr = ' + '.join([m.latex('') for m in self.monomials])
        return bracket + latexstr + bracket


    def __eq__(self, p):
        return (isinstance(p, Posynomial) and str(self) == str(p))

    def __ne__(self, x):
        return not self.__eq__(x)


    def __pow__(self, x):
        return Posynomial([m**x for m in self.monomials])

    def __div__(self, m):
        return Posynomial([m_/m for m_ in self.monomials])

    def __mul__(self, m):
        return Posynomial([m_*m for m_ in self.monomials])