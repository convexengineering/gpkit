from monomial import Monomial
from array import array

def vectify(s,n):
    assert len(s.split()) == 1, "Accepts only a single variable name."
    return array([Monomial(s+str(i)) for i in xrange(n)])

