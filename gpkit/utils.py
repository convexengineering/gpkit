from nomials import Monomial

def monify(s):
    """
    Turns a whitespace separated string into singlet monomials.
    """
    return [Monomial(x) for x in s.split()]