from collections import defaultdict
from nomials import Monomial


def sumlist(l, attr=None):
    if not attr:
        pile = l[0]
        for el in l[1:]:
            pile += el
    else:
        pile = getattr(l[0], attr)
        for el in l[1:]:
            pile += getattr(el, attr)
    return pile


def locate_vars(exps):
    var_locs = defaultdict(list)
    for i, exp in enumerate(exps):
        for var in exp:
            var_locs[var].append(i)
    return var_locs


def substitution(var_locs, exps, cs, substitutions, val=None):
    if val is not None and isinstance(substitutions, str):
        # singlet variable
        substitutions = {substitutions: val}

    subs, descrs = {}, {}
    for var, sub in substitutions.iteritems():
        if var in var_locs:
            try:
                # described variable
                assert len(sub) == 2 and isinstance(sub[1], str)
                subs[var] = sub[0]
                descrs[var] = sub[1]
            except:
                # regular variable
                subs[var] = sub
        else:
            try:
                # vector variable
                if all((isinstance(val, (int, float, Monomial))
                        for val in sub)):
                    # sub is a vector
                    vsub = [(var+str(j), val)
                            for (j, val) in enumerate(sub)]
                elif all((isinstance(val, (int, float, Monomial))
                          for val in sub[0])):
                    # sub's first element is a vector
                    vsub = [(var+str(j), val)
                            for (j, val) in enumerate(sub[0])]
                    assert isinstance(sub[1], str)
                    descrs[var] = sub[1]

                for var, val in vsub:
                    if var in var_locs:
                        subs[var] = val
            except: pass
    if not subs:
        return exps, cs, descrs, subs
    else:
        exps_ = [HashVector(exp) for exp in exps]
        cs_ = list(cs)
        for var, sub in subs.iteritems():
            for i in var_locs[var]:
                x = exps_[i][var]
                del exps_[i][var]
                if isinstance(sub, (int, float)):
                    # scalar substitution
                    cs_[i] *= sub**x
                elif isinstance(sub, str):
                    # variable name substitution
                    exps_[i] += HashVector({sub: x})
                elif isinstance(sub, Monomial):
                    # monomial substitution
                    exps_[i] += sub.exp*x
                    cs_[i] *= sub.c**x
        return exps_, cs_, descrs, subs


def invalid_types(oper, a, b):
    typea = a.__class__.__name__
    typeb = b.__class__.__name__
    raise TypeError("unsupported operand types"
                    " for %s: '%s' and '%s'" % (oper, typea, typeb))


class HashVector(dict):
    "A simple, sparse, string-indexed immutable vector."

    # unsettable and hashable
    def __hash__(self):
        if not hasattr(self, '_hashvalue'):
            self._hashvalue = hash(tuple(self.items()))
        return self._hashvalue

    def __setitem__(self, key, value):
        raise TypeError("HashVectors are immutable.")

    # standard element-wise arithmetic
    def __neg__(self):
        return HashVector({key: -val for (key, val) in self.iteritems()})

    def __pow__(self, x):
        if isinstance(other, (int, float)):
            return HashVector({key: val**x for (key, val) in self.iteritems()})
        else:
            invalid_types('** or pow()', self, x)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return HashVector({key: val*other
                               for (key, val) in self.iteritems()})
        elif isinstance(other, dict):
            keys = set(self.keys()).union(other.keys())
            sums = {key: self.get(key, 0) * other.get(key, 0)
                    for key in keys}
            return HashVector(sums)
        else:
            invalid_types('*', self, other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return HashVector({key: val+other
                               for (key, val) in self.iteritems()})
        elif isinstance(other, dict):
            keys = set(self.keys()).union(other.keys())
            sums = {key: self.get(key, 0) + other.get(key, 0)
                    for key in keys}
            return HashVector(sums)
        else:
            invalid_types('+', self, other)

    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return other + -self
    def __radd__(self, other): return self + other

    def __div__(self, other): return self * other**-1
    def __rdiv__(self, other): return other * self**-1
    def __rmul__(self, other): return self * other