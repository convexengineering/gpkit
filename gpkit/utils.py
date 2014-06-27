from nomials import Monomial
from matrix import matrix

def monify(s):
    """
    Turns a whitespace separated string into singlet monomials.
    """
    return [Monomial(x) for x in s.split()]

def vectify(s,n):
    assert len(s.split()) == 1, "Accepts only a single variable name."
    return matrix([Monomial(s+str(i)) for i in xrange(n)]).T