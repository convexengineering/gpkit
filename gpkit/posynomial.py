"""
Posynomial expression (term)
"""

# NOTE: there's a circular import at the bottom of this file.
#   since the monomial class only calls the Posynomial class
#   when there's addition or subtraction, it might be alright?

class Posynomial(object):
    def __init__(self, posynomials):

        monomials = []
        for p in posynomials:
            if isinstance(p, Posynomial):
                monomials += p.monomials
            elif isinstance(p, Monomial):
                monomials.append(p)
            else:
                # assume it's a number
                monomials.append(Monomial({}, p))

        # currently this removes duplicates!
        # simplifying need be done in monomial arithmetic
        # and in posynomial creation
        self.monomials = set(monomials)

        if len(self.monomials) <= 1:
            raise TypeError, "Need more than one monomial to make a posynomial"

        vars = []
        for m in self.monomials:
            vars += m.vars
        self.vars = set(vars)
    
    def __repr__(self):
        strlist = [str(m) for m in self.monomials]
        return ' + '.join(strlist)
    
    def latex(self, bracket='$'):
        latexstr = ' + '.join([m.latex('') for m in self.monomials])
        return bracket + latexstr + bracket

    def __hash__(self):
        return hash(str(self))


    def __eq__(self, p):
        return (isinstance(p, Posynomial) and str(self) == str(p))

    def __ne__(self, x):
        return not self.__eq__(x)


    def __pow__(self, x):
        return Posynomial([m**x for m in self.monomials])

    def __div__(self, m):
        return Posynomial([m_s/m for m_s in self.monomials])

    def __rdiv__(self, m):
        return Posynomial([m/m_s for m_s in self.monomials])

    def __mul__(self, m):
        return Posynomial([m_s*m for m_s in self.monomials])

    def __rmul__(self, m):
        return Posynomial([m*m_s for m_s in self.monomials])

    def __sub__(self, m):
        return Posynomial([self, -1*m])

    def __rsub__(self, m):
        return Posynomial([m, -1*self])

    def __add__(self, m):
        return Posynomial([self, m])

    def __radd__(self, m):
        return Posynomial([m, self])

    def __neg__(self):
        return Posynomial([-m_s for m_s in self.monomials])

from monomial import Monomial