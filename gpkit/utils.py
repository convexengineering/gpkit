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
    Turns a whitespace separated string into a list of singlet monomials.
    """
    return [Monomial(x) for x in s.split()]

def dict_monify(s):
	"""
    Turns a whitespace separated string into a dictionary of singlet monomials.
    Potentially useful to call as globals().update(dict_monify(s))
    """
	if isinstance(s, str):
		return {x: Monomial(x) for x in s.split()}
	else:
		return {x: Monomial(x) for x in s}

def monify_up(d, s):
	"""
	Updates a dictionary with a dict_monify call.
	Potentially useful to call with globals() for d
	"""
	return d.update(dict_monify(s))
