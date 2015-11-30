"""Machinery for exps, cs, varlocs data -- common to nomials and programs"""
import numpy as np

from collections import defaultdict

from functools import reduce as functools_reduce
from operator import add

from .varkey import VarKey
from .small_classes import HashVector, Quantity, KeySet, KeyDict
from .small_scripts import mag


class NomialData(object):
    """Object for holding cs, exps, and other basic 'nomial' properties.

    cs: array (coefficient of each monomial term)
    exps: tuple of {VarKey: float} (exponents of each monomial term)
    varlocs: {VarKey: list} (terms each variable appears in)
    units: pint.UnitsContainer
    """
    def __init__(self, exps=None, cs=None, simplify=True):
        if exps is None and cs is None:
            # pass through for classmethods to get a NomialData object,
            # which they will then call __init__ on
            return
        if simplify:
            exps, cs = simplify_exps_and_cs(exps, cs)
        self.exps, self.cs = exps, cs
        self.any_nonpositive_cs = any(mag(c) <= 0 for c in self.cs)

        varlocs = {}
        for i, exp in enumerate(exps):
            for var in exp:
                if var not in varlocs:
                    varlocs[var] = []
                varlocs[var].append(i)
        self.varlocs = varlocs
        self.varkeys = KeySet(self.varlocs)
        self.values = KeyDict({vk: vk.descr["value"] for vk in self.varkeys
                               if "value" in vk.descr})

        if isinstance(self.cs, Quantity):
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

    @classmethod
    def fromnomials(cls, nomials):
        """Construct a NomialData from an iterable of Signomial objects"""
        nd = cls()  # use pass-through version of __init__
        nd.init_from_nomials(nomials)
        return nd

    def init_from_nomials(self, nomials):
        """Way to initialize from nomials. Calls __init__.
        Used by subclass __init__ methods.
        """
        exps = functools_reduce(add, (tuple(s.exps) for s in nomials))
        cs = np.hstack((mag(s.cs) for s in nomials))
        # nomials are already simplified, so simplify=False
        NomialData.__init__(self, exps, cs, simplify=False)
        self.nomials = nomials  # TODO eliminate constructor-dependent state
        self.units = tuple(s.units for s in nomials)

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, hash(self))

    def sub(self, substitutions, val=None, require_positive=True):
        if hasattr(self, "nomials"):
            subbed_nomials = [n.sub(substitutions, val, require_positive)
                              for n in self.nomials]
            nd = NomialData.fromnomials(subbed_nomials)
        else:
            _, exps, cs, _ = substitution(self, substitutions, val)
            nd = NomialData(exps, cs)
        return nd

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
        var, = self.varkeys[var]
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
        units = Quantity(1, cs_[0].units)
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
