from gpkit.array import array
from gpkit import Monomial


def vectify(s, n):
    """
    Turns a string ('x') and a number (3) into an array ([x0 x1 x2])
    """
    assert len(s.split()) == 1, "Accepts only a single variable name."
    return array([Monomial(s+str(i)) for i in xrange(n)])


def monify(s):
    """
    Turns a whitespace separated string into singlet monomials.
    """
    return [Monomial(x) for x in s.split()]
