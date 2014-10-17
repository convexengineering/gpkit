# -*- coding: utf-8 -*-
"Module containing miscellanous useful functions"

from collections import defaultdict

from .nomials import Monomial


def locate_vars(exps):
    """From posynomial exponents, forms a dictionary
       of what monomials each variable is in."""
    var_locs = defaultdict(list)
    for i, exp in enumerate(exps):
        for var in exp:
            var_locs[var].append(i)
    return var_locs


def substitution(var_locs, exps, cs, substitutions, val=None):
    """Does efficient substituton into a list of monomials.

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
        descrs : dict
            Descriptions of substituted variables.
        subs_ : dict
            Substitutions to apply to the above.
    """
    if val is not None and isinstance(substitutions, (str, Monomial)):
        # singlet variable
        var = substitutions
        if isinstance(var, Monomial):
            if (var.c == 1 and len(var.exp) == 1
               and list(var.exp.values())[0] == 1):
                var = list(var.exp.keys())[0]
        substitutions = {var: val}

    subs, descrs = {}, {}
    for var, sub in substitutions.items():
        # substitute from singlet variable
        if isinstance(var, Monomial):
            if (var.c == 1 and len(var.exp) == 1
               and list(var.exp.values())[0] == 1):
                var = list(var.exp.keys())[0]
        if var in var_locs:
            try:
                # described variable
                assert not isinstance(sub, str)
                if len(sub) > 1 and isinstance(sub[-1], str):
                    if len(sub) > 2 and isinstance(sub[-2], str):
                        subs[var] = sub[0]
                        descrs[var] = sub[-2:]
                    else:
                        subs[var] = sub[0]
                        descrs[var] = [None, sub[-1]]
            except:
                # regular variable
                subs[var] = sub
        else:
            try:
                if all((isinstance(val, (int, float, Monomial))
                        for val in sub)):
                    # sub is a vector
                    vsub = [("{%s}_{%i}" % (var, j), val)
                            for (j, val) in enumerate(sub)]
                elif all((isinstance(val, (int, float, Monomial))
                          for val in sub[0])):
                    # sub's first element is a vector
                    vsub = [("{%s}_{%i}" % (var, j), val)
                            for (j, val) in enumerate(sub[0])]
                    # sub's last element is description
                    assert isinstance(sub[-1], str)
                    descrs[var] = sub[-1]
                # got a vector variable
                for var, val in vsub:
                    if var in var_locs:
                        subs[var] = val
            except: pass
    if not subs:
        return var_locs, exps, cs, descrs, subs
    else:
        exps_ = [HashVector(exp) for exp in exps]
        cs_ = list(cs)
        var_locs_ = defaultdict(list)
        var_locs_.update({var: list(idxs)
                          for (var, idxs) in var_locs.items()})
        for var, sub in subs.items():
            for i in var_locs[var]:
                x = exps_[i][var]
                del exps_[i][var]
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
        return var_locs_, exps_, cs_, descrs, subs


def invalid_types_for_oper(oper, a, b):
    "Raises TypeError for unsupported operations."
    typea = a.__class__.__name__
    typeb = b.__class__.__name__
    raise TypeError("unsupported operand types"
                    " for %s: '%s' and '%s'" % (oper, typea, typeb))


class HashVector(dict):
    """A simple, sparse, string-indexed immutable vector that inherits from dict.

    The HashVector class supports element-wise arithmetic.

    Any undeclared variables are assumed to have a value of zero.
    """

    # unsettable and hashable
    def __hash__(self):
        if not hasattr(self, "_hashvalue"):
            self._hashvalue = hash(tuple(self.items()))
        return self._hashvalue

    # TODO: implement this without breaking copy.deepcopy
    # def __setitem__(self, key, value):
    #     raise TypeError("HashVectors are immutable.")

    def __neg__(self):
        return HashVector({key: -val for (key, val) in self.items()})

    def __pow__(self, x):
        if isinstance(other, (int, float)):
            return HashVector({key: val**x for (key, val) in self.items()})
        else:
            invalid_types_for_oper("** or pow()", self, x)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return HashVector({key: val*other
                               for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self.keys()).union(other.keys())
            sums = {key: self.get(key, 0) * other.get(key, 0)
                    for key in keys}
            return HashVector(sums)
        else:
            invalid_types_for_oper("*", self, other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return HashVector({key: val+other
                               for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self.keys()).union(other.keys())
            sums = {key: self.get(key, 0) + other.get(key, 0)
                    for key in keys}
            return HashVector(sums)
        else:
            invalid_types_for_oper("+", self, other)

    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return other + -self
    def __radd__(self, other): return self + other

    def __div__(self, other): return self * other**-1
    def __rdiv__(self, other): return other * self**-1
    def __rmul__(self, other): return self * other
