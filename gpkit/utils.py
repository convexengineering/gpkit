from gpkit import PosyArray
from gpkit import Monomial


def vectify(s, n):
    "From a string ('x') and a number (3) returns an array ([x0 x1 x2])"
    return PosyArray([Monomial("{%s}_{%i}" % (name, i))
                      for i in xrange(length)])


def monify(s):
    "From a whitespace separated string, returns a list of singlet monomials."
    return [Monomial(x) for x in s.split()]


def dict_monify(s):
    "From a dictionary of name:description, returns one of monomials."
    monomial_dict = {}
    for var, val in s.iteritems():
        try:
            if val[0] == "vector":
                m = vectify(var, val[1])
                for el in m:
                    descr_var(el, val[2:])
            else:
                assert False
        except:
            m = Monomial(var)
            descr_var(m, val)
        monomial_dict.update({var: m})
    return monomial_dict


def descr_var(m, descr):
    var = m.exp.keys()[0]
    if isinstance(descr, str):
        m.var_descrs = {var: [None, descr]}
    else:
        try:
            if isinstance(descr[-1], str):
                if isinstance(descr[-2], str):
                    m.var_descrs = {var: descr[-2:]}
                else:
                    m.var_descrs = {var: [None, descr[-1]]}
        except (TypeError, IndexError):
            pass


def monify_up(d, s):
    """Updates the given dictionary with dict_monify(s).

    Quite useful in an interactive environment with globals() for d.
    """
    return d.update(dict_monify(s))
