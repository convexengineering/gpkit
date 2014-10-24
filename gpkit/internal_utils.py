# -*- coding: utf-8 -*-
"Module containing miscellanous useful functions"

from collections import defaultdict
from collections import Iterable

from .nomials import Monomial
from .nomials import VarKey
from .nomials import monovector


def locate_vars(exps):
    """From posynomial exponents, forms a dictionary
       of what monomials each variable is in."""
    var_locs = defaultdict(list)
    for i, exp in enumerate(exps):
        for var in exp:
            var_locs[var].append(i)
    return var_locs


def is_sweepvar(sub):
    "Determines if a given substitution indicates a sweep."
    try: return sub[0] == "sweep"
    except: return False


def vectorsub(subs, var, subiter, var_locs):
    "Vectorized substitution via monovectors and VarKeys."
    if isinstance(var, VarKey):
        var = monovector(**var.descr)
    if len(var) == len(subiter):
        for i in range(len(var)):
            v = VarKey(var[i])
            if v in var_locs:
                subs[v] = subiter[i]
    else:
        raise ValueError("tried substituting %s for %s, but their"
                         "lengths were unequal." % (sub, var))


def substitution(var_locs, exps, cs, substitutions, val=None):
    """Efficient substituton into a list of monomials.

        Parameters
        ----------
        var_locs : dict
            Dictionary of monomial indexes for each variable.
        exps : dict
            Dictionary of variable exponents for each monomial.
        cs : list
            Coefficients each monomial.
        substitutions : dict
            Substitutions to apply to the above.
        val : number (optional)
            Used to substitute singlet variables.

        Returns
        -------
        var_locs_ : dict
            Dictionary of monomial indexes for each variable.
        exps_ : dict
            Dictionary of variable exponents for each monomial.
        cs_ : list
            Coefficients each monomial.
        subs_ : dict
            Substitutions to apply to the above.
    """

    if val is not None:
        substitutions = {substitutions: val}

    subs = {}
    for var, sub in substitutions.items():
        if isinstance(sub, Iterable):
            if len(sub) == 1:
                sub = sub[0]
        if is_sweepvar(sub):
            pass
        elif isinstance(sub, Iterable):
            vectorsub(subs, var, sub, var_locs)
        elif VarKey(var) in var_locs:
            subs[VarKey(var)] = sub

    if not subs:
        raise KeyError("could not find anything to substitute.")

    exps_ = [HashVector(exp) for exp in exps]
    cs_ = list(cs)
    var_locs_ = defaultdict(list)
    var_locs_.update({var: list(idxs) for (var, idxs) in var_locs.items()})
    for var, sub in subs.items():
        for i in var_locs[var]:
            x = exps_[i].pop(var)
            var_locs_[var].remove(i)
            if not var_locs_[var]:
                del var_locs_[var]
            if isinstance(sub, (int, float)):
                # scalar substitution
                cs_[i] *= sub**x
            elif isinstance(sub, str):
                # variable name substitution
                exps_[i] += HashVector({sub: x})
                var_locs_[sub].append(i)
            elif isinstance(sub, Monomial):
                # monomial substitution
                exps_[i] += sub.exp*x
                cs_[i] *= sub.c**x
                for subvar in sub.exp:
                    var_locs_[subvar].append(i)
    return var_locs_, exps_, cs_, subs


def invalid_types_for_oper(oper, a, b):
    "Raises TypeError for unsupported operations."
    typea = a.__class__.__name__
    typeb = b.__class__.__name__
    raise TypeError("unsupported operand types"
                    " for %s: '%s' and '%s'" % (oper, typea, typeb))


class HashVector(dict):
    """A simple, sparse, string-indexed immutable vector that inherits from dict.

    The HashVector class supports element-wise arithmetic:
    any undeclared variables are assumed to have a value of zero.

    Parameters
    ----------
    arg : iterable

    Example
    -------
    >>> x = gpkit.nomials.VarKey('x')
    >>> exp = gpkit.internal_utils.HashVector({x: 2})
    """

    def __hash__(self):
        "Allows HashVectors to be used as dictionary keys."
        if not hasattr(self, "_hashvalue"):
            self._hashvalue = hash(tuple(self.items()))
        return self._hashvalue

    def __neg__(self):
        "Return Hashvector with each value negated."
        return HashVector({key: -val for (key, val) in self.items()})

    def __pow__(self, x):
        "Accepts scalars. Return Hashvector with each value put to a power."
        if isinstance(other, (int, float)):
            return HashVector({key: val**x for (key, val) in self.items()})
        else:
            invalid_types_for_oper("** or pow()", self, x)

    def __mul__(self, other):
        """Accepts scalars and dicts. Returns with each value multiplied.

        If the other object inherits from dict, multiplication is element-wise
        and their key's intersection will form the new keys."""
        if isinstance(other, (int, float)):
            return HashVector({key: val*other for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self.keys()).intersection(other.keys())
            sums = {key: self[key] * other[key] for key in keys}
            return HashVector(sums)
        else:
            invalid_types_for_oper("*", self, other)

    def __add__(self, other):
        """Accepts scalars and dicts. Returns with each value added.

        If the other object inherits from dict, addition is element-wise
        and their key's union will form the new keys."""
        if isinstance(other, (int, float)):
            return HashVector({key: val+other
                               for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self.keys()).union(other.keys())
            sums = {key: self.get(key, 0) + other.get(key, 0) for key in keys}
            return HashVector(sums)
        else:
            invalid_types_for_oper("+", self, other)

    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return other + -self
    def __radd__(self, other): return self + other

    def __div__(self, other): return self * other**-1
    def __rdiv__(self, other): return other * self**-1
    def __rmul__(self, other): return self * other
