from gpkit import array
from gpkit import Monomial


def vectify(s, n):
    "From a string ('x') and a number (3) returns an array ([x0 x1 x2])"
    return array([Monomial("%s_%i" % (s, i)) for i in xrange(n)])


def monify(s):
    "From a whitespace separated string, returns a list of singlet monomials."
    return [Monomial(x) for x in s.split()]


def dict_monify(s):
    "From a dictionary of name:description, returns one of monomials."
    monomial_dict = {}
    for var, val in s.iteritems():
        m = Monomial(var)
        if isinstance(val, str):
            m.var_descrs = {var: [None, val]}
        else:
            try:
                if isinstance(val[-1], str):
                    if isinstance(val[-2], str):
                        m.var_descrs = {var: val[-2:]}
                    else:
                        m.var_descrs = {var: [None, val[-1]]}
            except (TypeError, IndexError):
                pass
        monomial_dict.update({var: m})
    return monomial_dict


def monify_up(d, s):
    """Updates the given dictionary with dict_monify(s).

    Quite useful in an interactive environment with globals() for d.
    """
    return d.update(dict_monify(s))
