"""Machinery for exps, cs, varlocs data -- common to nomials and programs"""
import numpy as np

from collections import defaultdict

from functools import reduce as functools_reduce
from operator import add

from .varkey import VarKey
from .small_classes import HashVector, Quantity
from .small_scripts import mag


class NomialData(object):
    """Object for holding cs, exps, and other basic 'nomial' properties.

    cs: array (coefficient of each monomial term)
    exps: tuple of {VarKey: float} (exponents of each monomial term)
    varlocs: {VarKey: list} (terms each variable appears in)
    units: pint.UnitsContainer
    """
    def __init__(self, exps=None, cs=None, nomials=None, simplify=True):
        if nomials and (exps or cs):
            raise ValueError("The NomialData initializor accepts either"
                             " exps and cs, or nomials, but not both.")
        elif nomials:
            exps = functools_reduce(add, (tuple(s.exps) for s in nomials))
            cs = np.hstack((mag(s.cs) for s in nomials))
            simplify = False  # nomials have already been simplified
        elif exps is None or cs is None:
            raise ValueError("creation of a NomialData requires exps and cs.")

        if simplify:
            exps, cs = simplify_exps_and_cs(exps, cs)
        self.exps, self.cs = exps, cs
        self.any_nonpositive_cs = any(mag(c) <= 0 for c in self.cs)
        self.varlocs, self.varstrs = locate_vars(self.exps)
        self.values = {vk: vk.descr["value"] for vk in self.varlocs
                       if "value" in vk.descr}
        if nomials:
            self.units = tuple(s.units for s in nomials)
        elif isinstance(self.cs, Quantity):
            self.units = Quantity(1, self.cs.units)
        else:
            self.units = None

        self._hashvalue = None

    def __hash__(self):
        if self._hashvalue is None:
            # confirm lengths before calling zip
            assert len(self.exps) == len(self.cs)
            self._hashvalue = hash(tuple(zip(self.exps, self.cs)))
        return self._hashvalue

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, hash(self))

    def _get_varkey(self, var):
        """Cast a var associated with this problem to type VarKey

        Arguments
        ---------
        var (Variable, VarKey, or str):
            the variable to cast

        Returns
        -------
        VarKey
        """
        if isinstance(var, VarKey):
            return var
        if var in self.varstrs:
            return self.varstrs[var]
        try:
            return var.varkey
        except AttributeError:
            raise TypeError("Cannot convert %s to VarKey" % var)

    def diff(self, var):
        """Derivative of this with respect to a Variable

        Arguments
        ---------
        var (Variable):
            Variable to take derivative with respect to

        Returns
        -------
        NomialData
        """
        var = self._get_varkey(var)
        exps, cs = [], []
        # var.units may be str if units disabled
        var_units = (var.units if var.units and not isinstance(var.units, str)
                     else 1)
        for i, exp in enumerate(self.exps):
            exp = HashVector(exp)   # copy -- exp is mutated below
            e = exp.get(var, 0)
            if var in exp:
                exp[var] -= 1
            exps.append(exp)
            cs.append(e*self.cs[i] / var_units)
        # don't simplify to keep length same as self
        return NomialData(exps=exps, cs=cs, simplify=False)

    def __eq__(self, other):
        """Equality test"""
        if not all(hasattr(other, a) for a in ("exps", "cs", "units")):
            return NotImplemented
        if self.exps != other.exps:
            return False
        if not all(mag(self.cs) == mag(other.cs)):
            return False
        if self.units != other.units:
            return False
        return True


def simplify_exps_and_cs(exps, cs, return_map=False):
    """Reduces the number of monomials, and casts them to a sorted form.

    Arguments
    ---------

    exps : list of Hashvectors
        The exponents of each monomial
    cs : array of floats or Quantities
        The coefficients of each monomial
    return_map : bool (optional)
        Whether to return the map of which monomials combined to form a
        simpler monomial, and their fractions of that monomial's final c.

    Returns
    -------
    exps : list of Hashvectors
        Exponents of simplified monomials.
    cs : array of floats or Quantities
        Coefficients of simplified monomials.
    mmap : list of HashVectors
        List for each new monomial of {originating indexes: fractions}
    """
    matches = defaultdict(float)
    if return_map:
        expmap = defaultdict(dict)
    for i, exp in enumerate(exps):
        exp = HashVector({var: x for (var, x) in exp.items() if x != 0})
        matches[exp] += cs[i]
        if return_map:
            expmap[exp][i] = cs[i]

    if len(matches) > 1:
        zeroed_terms = (exp for exp, c in matches.items() if mag(c) == 0)
        for exp in zeroed_terms:
            del matches[exp]

    exps_ = tuple(matches.keys())
    cs_ = list(matches.values())
    if isinstance(cs_[0], Quantity):
        units = cs_[0]/cs_[0].magnitude
        cs_ = [c.to(units).magnitude for c in cs_] * units
    else:
        cs_ = np.array(cs_, dtype='float')

    if not return_map:
        return exps_, cs_
    else:
        mmap = [HashVector() for c in cs_]
        for i, item in enumerate(matches.items()):
            exp, c = item
            for j in expmap[exp]:
                mmap[i][j] = mag(expmap[exp][j]/c)
        return exps_, cs_, mmap


def locate_vars(exps):
    "From exponents form a dictionary of which monomials each variable is in."
    varlocs = defaultdict(list)
    varstrs = defaultdict(set)
    for i, exp in enumerate(exps):
        for var in exp:
            varlocs[var].append(i)
            varstrs[var.name].add(var)

    varkeys_ = dict(varstrs)
    for name, varl in varkeys_.items():
        for vk in varl:
            descr = vk.descr
            break
        if "shape" in descr:
            # vector var
            newlist = np.zeros(descr["shape"], dtype="object")
            for var in varl:
                newlist[var.descr["idx"]] = var
            varstrs[name] = newlist
        else:
            if len(varl) == 1:
                varstrs[name] = varl.pop()
            else:
                varstrs[name] = []
                for var in varl:
                    if "model" in var.descr:
                        varstrs[name+"_%s" % var.descr["model"]] = var
                    else:
                        varstrs[name].append(var)
                if len(varstrs[name]) == 1:
                    varstrs[name] = varstrs[name][0]
                elif len(varstrs[name]) == 0:
                    del varstrs[name]

    return dict(varlocs), dict(varstrs)
