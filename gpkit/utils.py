from nomials import Monomial

def monify(s):
    """
    Turns a whitespace separated string into singlet monomials.
    """
    return [Monomial(x) for x in s.split()]

from matrix import matrix

def vectify(s,n):
    assert len(s.split()) == 1, "Accepts only a single variable name."
    return matrix([Monomial(s+str(i)) for i in xrange(n)]).T
