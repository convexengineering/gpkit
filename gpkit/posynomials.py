from collections import defaultdict
from collections import namedtuple
from itertools import chain


class Posynomial(object):
    """A representation of a Posynomial

    attributes:
        c: a list of monomial coefficients
        terms: a list of dicionaries of monomial exponents
        exps: a dictionary of (term_idx, exponent_value) for each variable
    """

    def __init__(self, arg, c=None):
        if isinstance(arg, (int, float)):
            # build scalar
            c = [c] if c else [1]
            terms = [{}]

        elif isinstance(arg, str):
            # build simple Monomial
            c = [c] if c else [1]
            terms = [{arg: 1}]

        elif isinstance(arg, list):
            # build Posynomial from list of Posynomials
            if c is not None:
                raise ValueError("c should not be declared when creating"
                                 " a Posynomial from a list of Posynomials!")
            c, terms = [], []
            for idx, item in enumerate(arg):
                if isinstance(item, Monomial):
                    c.append(item.c)
                    terms.append(item.terms)
                elif isinstance(item, (int, float)):
                    c.append(item)
                    terms.append({})
                elif isinstance(item, Posynomial):
                    c += item.c
                    terms += item.terms
                else:
                    raise ValueError("Invalid Posynomial: %s" % item)

        elif isinstance(arg, dict):
            arg_values_are_numbers = (isinstance(val, (int, float))
                                      for val in arg.values())
            if all(arg_values_are_numbers):
                # build Monomial from term dictionary
                c = [c] if c else [1]
                terms = [arg]
            else:
                # build Posynomial from exps dictionary
                exps = arg
                c = c if c else [1]*len(exps)
                terms = [defaultdict(float)]*len(c)
                for var, exp in exps:
                    term_idx, value = exp
                    terms[term_idx][var] = value

        self.c, self.terms, self.exps = simplify(c, terms)
        self.vars = set(self.exps)  # backwards compatibility

        if len(self.terms) == 1:
            self.__class__ = Monomial
            self.c = c[0]
            if c <= 0:
                raise ValueError("Monomials must have positive coefficients!")
            self.terms = self.terms[0]
            self.exps = self.terms
            self.monomials = [self]
        else:
            self.monomials = [Monomial(term, c_)
                              for (term, c_) in zip(terms, c)]

    def __repr__(self):
        return ' + '.join((str(m) for m in self.monomials))

    def latex(self, bracket='$'):
        latexstr = ' + '.join((m.latex(bracket='')
                               for m in self.monomials))
        return bracket + latexstr + bracket

    def __eq__(self, other):
        if isinstance(other, Posynomial):
            return (self.exps == other.exps
                    and self.c == other.c
                    and self.__class__ == other.__class__)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        "Treat exclusive inequalities as inclusive ones."
        return self <= other

    def __le__(self, other):
        "Overload inequality operators to return a posynomial constraint"
        return self / other

    def __div__(self, other):
        if not isinstance(other, (Monomial, int, float)):
            raise TypeError("Posynomials may only be divided"
                            " by Monomials and scalars")
        return Posynomial([s * other**-1 for s in self.monomials])

    def __pow__(self, x):
        if not isinstance(x, int) and x > 1:
            raise TypeError("Posynomials may only be raised"
                            " to positive integers")
        p = self
        while x > 1:
            p *= self
            x -= 1
        return p

    def __add__(self, other):
        if isinstance(other, (int, float, Posynomial)):
            return Posynomial([self, other])
        else:
            raise TypeError("Invalid Posynomial")

    def __mul__(self, other):
        if isinstance(other, (Monomial, int, float)):
            # use the __mul__ defined in Monomial etc.
            return Posynomial([m_s * other for m_s in self.monomials])
        elif isinstance(other, Posynomial):
            monomials = []
            for m_s in self.monomials:
                for m_o in other.monomials:
                    monomials.append(m_s*m_o)
            return Posynomial(monomials)
        else:
            raise TypeError("Posynomials may only be multiplied"
                            " by other Posynomials and by scalars")

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


class Monomial(Posynomial):
    def __rdiv__(self, m):
        return m * self**-1

    def __pow__(self, x):
        term = {var: val*x for var, val in self.exps.iteritems()}
        c = self.c**x
        return Monomial(term, c)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Monomial(self.exps, other * self.yc)
        elif isinstance(other, Monomial):
            allvars = set().union(self.exps, other.exps)
            c = self.c * other.c
            term = {var: self.exps[var] + other.exps[var]
                    for var in allvars}
            return Monomial(term, c)
        elif isinstance(other, Posynomial):
            return other * self
        else:
            raise TypeError("Monomials may only be multiplied"
                            " by other Posynomials and by scalars")

    def __string(self, mult_symbol='*'):
        varstrs = ['%s^%g' % (var, exp) if exp != 1 else var
                   for (var, exp) in self.exps.iteritems()]
        cstrs = ["%g" % self.c] if self.c != 1 or not varstrs else []
        return mult_symbol.join(cstrs + varstrs)

    def __repr__(self):
        return self.__string()

    def latex(self, bracket='$'):
        latexstr = self.__string(mult_symbol='')
        # could multiply with a space?
        return bracket + latexstr + bracket


def simplify(c, terms):
    "Pops any monomials that sum to another monomial and appends those sums"
    newc = defaultdict(float)
    term_hashes = [tuple(term.items()) for term in terms]
    for value, term, term_hash in zip(c, terms, term_hashes):
            newc[term_hash] += value

    c_, terms_ = [], []
    seen = set()
    for term, term_hash in zip(terms, term_hashes):
        if term_hash not in seen:
            seen.add(term_hash)
            c_.append(newc[term_hash])
            cleanterm = defaultdict(float,
                                    {var: val
                                     for (var, val) in term.iteritems()
                                     if val != 0})
            terms_.append(cleanterm)

    exps = defaultdict(list)
    for term_idx, term in enumerate(terms_):
        for var, value in term.iteritems():
            exps[var].append((term_idx, value))

    return c_, terms_, exps
