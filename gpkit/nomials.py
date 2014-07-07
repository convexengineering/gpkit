from collections import defaultdict
from itertools import chain
import numpy as np

# how to do read-only attributes in python?


def sort_and_simplify(exps, cs):
    "Reduces the number of monomials, and casts them to a sorted form."
    matches = defaultdict(float)
    for i, exp in enumerate(exps):
        exp = HashVector({var: x for (var, x) in exp.iteritems() if x != 0})
        matches[exp] += cs[i]
    return tuple(matches.keys()), tuple(matches.values())


class Posynomial(object):

    def __init__(self, exps, cs=1, *var_descrs):
        if isinstance(cs, (int, float)) and isinstance(exps, (dict, str)):
            if isinstance(exps, str):
                exps = {exps: 1}
            cs = [cs]
            exps = [exps]
        elif isinstance(exps, Posynomial):
            cs = exps.cs
            var_descrs = exps.var_descrs
            exps = exps.exps
        else:
            try: assert len(cs) == len(exps)
            except:
                raise TypeError("cs and exps must have the same length.")

        exps, cs = sort_and_simplify(exps, cs)
        if any((c <= 0 for c in cs)):
            raise ValueError("each c must be positive.")
        elif any((any((len(var.split()) > 1 for var in exp)) for exp in exps)):
            raise ValueError("variable names may not contain spaces.")

        self.exps = exps
        self.cs = cs
        self.var_locs = locate_vars(exps)
        if len(exps) == 1:
            self.__class__ = Monomial
            self.exp = exps[0]
            self.c = cs[0]

        self.var_descrs = defaultdict(str)
        var_descrs = list(chain(var_descrs))
        if var_descrs:
            if len(self.var_locs) == 1 and isinstance(var_descrs[0], str):
                self.var_descrs = {self.var_locs.keys()[0]: var_descrs[0]}
            else:
                for var_descr in var_descrs:
                    if isinstance(var_descr, dict):
                        self.var_descrs.update(var_descr)
                    else:
                        raise ValueError("invalid variable descriptions: "
                                         + str(var_descr))

    def sub(self, substitutions, val=None):
        exps, cs, newdescrs, subs = substitution(self.var_locs, self.exps, self.cs,
                                                 substitutions, val)
        return Posynomial(exps, cs, self.var_descrs, newdescrs)

    # hashing, immutability, Posynomial equality
    def __hash__(self):
        if not hasattr(self, '_hashvalue'):
            self._hashvalue = hash(tuple(self.exps, tuple(self.cs)))
        return self._hashvalue

    def __eq__(self, other):
        if isinstance(other, Posynomial):
            return (self.exps == other.exps and self.cs == other.cs)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    # inequality constraint generation
    def __le__(self, other):
        return self / other

    def __ge__(self, other):
        return other / self

    def __lt__(self, other):
        invalid_types('<', self, other)

    def __gt__(self, other):
        invalid_types('>', self, other)

    # string translations
    def _string(self, mult_symbol='*'):
        mstrs = []
        for c, exp in zip(self.cs, self.exps):
            varstrs = ['%s^%.2g' % (var, x) if x != 1 else var
                       for (var, x) in exp.iteritems() if x != 0]
            cstr = ["%.2g" % c] if c != 1 or not varstrs else []
            mstrs.append(mult_symbol.join(cstr+varstrs))
        return ' + '.join(mstrs)

    def __repr__(self):
        return self.__class__.__name__+'('+self._string()+')'

    def latex(self, bracket='$'):
        return bracket + self._string(mult_symbol='') + bracket

    # posynomial arithmetic
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Posynomial(self.exps+({},), self.cs+(other,), self.var_descrs)
        elif isinstance(other, Posynomial):
            return Posynomial(self.exps+other.exps, self.cs+other.cs,
                              self.var_descrs, other.var_descrs)
        else:
            invalid_types('+', self, other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Posynomial(self.exps, other*np.array(self.cs),
                              self.var_descrs)
        elif isinstance(other, Posynomial):
            C = np.outer(self.cs, other.cs)
            Exps = np.empty((len(self.exps), len(other.exps)), dtype='object')
            for i, exp_s in enumerate(self.exps):
                for j, exp_o in enumerate(other.exps):
                    Exps[i,j] = exp_s + exp_o
            return Posynomial(Exps.flatten(), C.flatten(),
                              self.var_descrs, other.var_descrs)
        else:
            invalid_types('*', self, other)

    def __rmul__(self, other): return self * other

    def __div__(self, other):
        if isinstance(other, Posynomial):
            if self.exps == other.exps:
                div_cs = np.array(self.cs)/np.array(other.cs)
                if all(div_cs == div_cs[0]):
                    return Monomial({}, div_cs[0])
        if isinstance(other, (int, float)):
            return Posynomial(self.exps, np.array(self.cs)/other)
        elif isinstance(other, Monomial):
            exps = [exp-other.exp for exp in self.exps]
            return Posynomial(exps, np.array(self.cs)/other.c,
                              self.var_descrs, other.var_descrs)
        else:
            invalid_types('/', self, other)

    def __pow__(self, x):
        if isinstance(x, int):
            if x > 1:
                p = 1
                while x > 0:
                    p *= self
                    x -= 1
                return p
            else:
                raise ValueError("Posynomials are only closed under"
                                 " positive integer exponents.")
        else:
            invalid_types('** or pow()', self, x)


class Monomial(Posynomial):

    def __rdiv__(self, other):
        if isinstance(other, (int, float, Posynomial)):
            return other * self**-1
        else:
            invalid_types('/', other, self)

    def __pow__(self, other):
        if isinstance(other, (int,float)):
            return Monomial(self.exp*other, self.c**other, self.var_descrs)
        else:
            invalid_types('** or pow()', self, x)

from helpers import *
