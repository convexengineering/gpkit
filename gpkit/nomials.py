from collections import defaultdict
from collections import Iterable

class nomial(object):
    ''' That which the nomials have in common '''
    # comparison
    def __ne__(self, m): return not self == m

    # GP constraint-making
    def __le__(self, m): return self / m
    def __ge__(self, m): return m / self
    def __lt__(self, m): return self <= m
    def __gt__(self, m): return self >= m

    # operators
    # __mul__ is defined by each nomial a little differently
    def __rmul__(self, m): return self * m

    def __add__(self, m):
        if self.monomial_match(m):
            return Monomial(self.exps, self.c + m.c)
        else:
            return Posynomial([self, m])

    def __radd__(self, m): return self + m
    def __sub__(self, m): return self + -m
    def __rsub__(self, m): return m + -self

    def monomial_match(self, m):
        if isinstance(self, Monomial) and isinstance(m, Monomial):
            both_scalar = self.is_scalar() and m.is_scalar()
            if both_scalar or self.exps == m.exps:
                return True
        else:
            return False


class Monomial(nomial):
    # hashing and representation
    def __hash__(self): return hash((self.c, self.eid))
    def __repr__(self): return self._str_tokens()
    def is_scalar(self):
        return all([e == 0 for e in self.exps.values()])

    # operators
    # __pow__ is defined below
    def __neg__(self): return Monomial(self.exps, -self.c)
    # __mul__ is defined below
    def __div__(self, m): return self * m**-1
    def __rdiv__(self, m): return m * self**-1

    def __eq__(self, m): return (isinstance(m, Monomial)
                                 and self.eid == m.eid
                                 and self.c == m.c)

    def __init__(self, exps, c=1):
        if isinstance(exps, str):
            exps = {exps: 1}
        # self.c: the monomial coefficent. needs to be positive.
        self.c = float(c)
        if not self.c > 0:
            raise ValueError('c must be positive')
        # self.exps: the exponents lookup table
        self.exps = defaultdict(int,
                                [(k,v) for (k,v) in exps.iteritems()
                                 if v != 0])
        # self.vars: the list of unique variables in a monomial
        self.vars = frozenset(self.exps.keys())
        # self.eid: effectively a hash of the exponents
        self.eid = hash(tuple(sorted(self.exps.items())))
        # self.monomials: makes combined mono- / posy-nomial lists nicer
        self.monomials = frozenset([self])

    def _str_tokens(self, joiner='*'):
        t = []
        for var in self.vars:
            exp = self.exps[var]
            if exp != 0:
                t.append('%s^%g' %
                         (var, exp) if exp != 1 else var)
        c = ["%g" % self.c] if self.c != 1 or not t else []
        return joiner.join(c + t)

    def latex(self, bracket='$'):
        latexstr = self._str_tokens('')  # could put a space in here?
        return bracket + latexstr + bracket

    def __pow__(self, x):
        exps = {var: x*self.exps[var] for var in self.vars}
        c = self.c**x
        return Monomial(exps, c)

    def __mul__(self, m):
        if not isinstance(m, Monomial):
            return Monomial(self.exps, self.c * m)
        else:
            allvars = frozenset().union(self.vars, m.vars)
            c = self.c * m.c
            exps = {var: self.exps[var] + m.exps[var]
                    for var in allvars}
            return Monomial(exps, c)

    def sub(self, constants_):
        constants = dict(constants_)
        # for vector-valued constants
        for var, constant in constants_.iteritems():
            if isinstance(constant, Iterable):
                del constants[var]
                for i, val in enumerate(constant):
                    constants[var+str(i)] = val

        overlap = self.vars.intersection(constants)
        if overlap:
            c = self.c
            exps = {var: exp
                    for var, exp in self.exps.iteritems()
                    if not var in overlap}
            for var in overlap:
                if isinstance(constants[var], (int,float)):
                    c *= constants[var]**self.exps[var]
            return Monomial(exps, c)
        else:
            return self


class Posynomial(nomial):
    # __pow__ is defined below
    # __neg__ is defined below
    # __mul__ is defined below
    # __div__ is defined below
    def __hash__(self): return hash(self.monomials)

    def __eq__(self, m): return (isinstance(m, self.__class__)
                                 and self.monomials == m.monomials)

    def __init__(self, posynomials):
        monomials = []
        for p in posynomials:
            monomials += list(p.monomials
                              if hasattr(p, 'monomials')
                              else [Monomial({}, p)])  # assume it's a number
        monomials = simplify(monomials)
        # self.monomials: the set of all monomials in the posynomial
        self.monomials = frozenset(monomials)

        loststr = "Some monomials did not simplify properly!"
        assert len(monomials) == len(self.monomials), loststr

        minlenstr = "Need at least one monomial to make a posynomial"
        assert len(self.monomials) > 0, minlenstr
        # TODO: return a Monomial if there's only one monomial
        # see the newnomials pull request for one attempt

        # self.vars: the set of all variables in the posynomial
        self.vars = frozenset().union(*[m.vars for m in self.monomials])

    def __repr__(self):
        strlist = [str(m) for m in self.monomials]
        return ' + '.join(strlist)

    def latex(self, bracket='$'):
        latexstr = ' + '.join([m.latex('') for m in self.monomials])
        return bracket + latexstr + bracket

    def __pow__(self, x):
        nota_bene = ("Posynomials are only closed when raised"
                     "to positive integers, not to %s" % x)
        assert isinstance(x, int) and x > 1, nota_bene
        p = 1
        while x > 0:
            p *= self
            x -= 1
        return p

    def __div__(self, m):
        if isinstance(m, Posynomial):
            raise TypeError("Posynomials are not closed under division")
        # assume monomial or number
        return Posynomial([s / m for s in self.monomials])

    def __rdiv__(self, m):
        raise TypeError("Posynomials are not closed under division")

    def __mul__(self, m):
        if isinstance(m, Posynomial):
            monoms = []
            for s in self.monomials:
                for m_ in m.monomials:
                    monoms.append(s*m_)
            return Posynomial(monoms)
        else:
            # assume monomial or number
            return Posynomial([s * m for s in self.monomials])

    def __neg__(self):
        return Posynomial([-m_s for m_s in self.monomials])

    def sub(self, constants):
        return Posynomial([m.sub(constants) for m in self.monomials])


def simplify(monomials):
    """ Bundles matching monomials from a list. """
    eidtable = defaultdict(list)
    for m_idx, m in enumerate(monomials):
        eidtable[m.eid].append(m_idx)
    dupes = [v for v in eidtable.itervalues() if len(v) != 1]

    if not dupes:
        return monomials
    else:
        dupe_idxs = []
        mout = []
        for idxs in dupes:
            dupe_idxs += idxs
            pile = monomials[idxs[0]]
            for m_idx in idxs[1:]:
                pile += monomials[m_idx]
            mout.append(pile)
        mout += [monomials[i] for i in xrange(len(monomials))
                 if not i in dupe_idxs]
        return mout
