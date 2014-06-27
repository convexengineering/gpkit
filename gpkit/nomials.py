class nomial(object):
    ''' That which the nomials have in common '''
    __hash__ = lambda self: hash(str(self))

    # comparison
    __eq__ = lambda self, m: (isinstance(m, self.__class__)
                              and str(self) == str(m))
    __ne__ = lambda self, m: not self == m

    # GP constraint-making
    __le__ = lambda self, m: self / m
    __lt__ = lambda self, m: self <= m

    # operators
    #__mul__ is defined by each nomial a little differently
    __rmul__ = lambda self, m: self * m
    #__add__ is defined below
    __radd__ = lambda self, m: self + m
    __sub__  = lambda self, m: self + -m
    __rsub__ = lambda self, m: m + -self

    def __add__(self, m):
        if self.monomial_match(m):
            return Monomial(self.exps, self.c + m.c)
        else:
            return Posynomial([self, m])

    def monomial_match(self, m):
        if isinstance(self, Monomial) and isinstance(m, Monomial):
            both_scalar = self.is_scalar() and m.is_scalar()
            return both_scalar or self.exps == m.exps
        else: 
            return False


class Monomial(nomial):
    # hashing and representation
    __repr__ = lambda self: self._str_tokens()
    is_scalar = lambda self: all([e==0 for e in self.exps.values()])

    # operators
    #__pow__ is defined below
    __neg__  = lambda self: Monomial(self.exps, -self.c)
    #__div__ is defined below
    __rdiv__ = lambda self, m: m * self ** -1
    __mul__  = lambda self, m: self / (m ** -1)
    #__sub__ is defined below


    def __init__(self, _vars, c=1, a=None):
        if isinstance(_vars, str):
            _vars = [_vars]
        # self.c: the monomial coefficent
        self.c = float(c)
        # self.vars: the list of unique variables in a monomial
        self.vars = frozenset(_vars)

        # self.exps: the exponents lookup table
        if isinstance(_vars, dict):
            self.exps = _vars
        else:
            N = len(_vars)
            if a is None: 
                a = [1] * N
            else:
                assert N == len(a), 'N=%s but len(a)=%s' % (N, len(a))
            self.exps = dict(zip(_vars, a))

        # self.eid: effectively a hash of the exponents
        self.eid = hash(tuple(sorted(m.exps.items())))

        # self.monomials: to make combined monomial / posynomial lists
        #                 easier to deal with
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
        a = [x * self.exps[var] for var in self.vars]
        c = self.c ** x
        return Monomial(self.vars, c, a)

    def __div__(self, m):
        if not isinstance(m, Monomial):
            return Monomial(self.exps, self.c / m)
        else:
            _vars = frozenset().union(self.vars, m.vars)
            c = self.c / m.c
            a = [self.exps.get(var, 0) - m.exps.get(var, 0)
                 for var in _vars]
            return Monomial(_vars, c, a)



class Posynomial(nomial):
    #__pow__ is defined below
    #__neg__ is defined below
    #__mul__ is defined below
    #__div__ is defined below

    def __init__(self, posynomials):
        monomials = []
        for p in posynomials:
            monomials += list(p.monomials 
                                if hasattr(p, 'monomials')
                                else [Monomial({}, p)]) # assume it's a number
        monomials = simplify(monomials)
        # self.monomials: the set of all monomials in the posynomial
        self.monomials = frozenset(monomials)

        loststr = "Some monomials did not simplify properly!"
        assert len(monomials) == len(self.monomials), loststr

        minlenstr = "Need more than one monomial to make a posynomial"
        assert len(self.monomials) >= 1, minlenstr

        # self.vars: the set of all variables in the posynomial
        self.vars = frozenset([m.vars for m in self.monomials])

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
        nota_bene = "Posynomials are not closed under division" 
        assert not isinstance(m, Posynomial), nota_bene
        # assume monomial or number
        return Posynomial([s / m for s in self.monomials])

    def __rdiv__(self, m):
        nota_bene = "Posynomials are not closed under division" 
        assert not isinstance(m, Posynomial), nota_bene
        # assume monomial or number
        return Posynomial([m / s for s in self.monomials])

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


def simplify(monomials):
    """ Bundles matching monomials from a list. """
    seen = []
    dupe_idxs = []
    dupes = {}
    for m_idx, m in enumerate(monomials):
        if m.eid in seen:
            if not dupes.get(m.eid):
                first_idx  = seen.index(m.eid)
                dupes[m.eid] = [first_idx]
                dupe_idxs.append(first_idx)

            dupes[m.eid].append(m_idx)
            dupe_idxs.append(m_idx)

        seen.append(m.eid)

    if not dupes:
        return monomials
    else:
        mout = [monomials[i] for i in xrange(len(monomials))
                                   if not i in dupe_idxs]
        for idxs in dupes.values():
            pile = monomials[idxs[0]]
            for m_idx in idxs[1:]:
                pile += monomials[m_idx]
            mout.append(pile)
        return mout

