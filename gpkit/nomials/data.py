"""Machinery for exps, cs, varlocs data -- common to nomials and programs"""
from collections import defaultdict
from functools import reduce as functools_reduce
from operator import add
import numpy as np
from ..small_classes import HashVector, Quantity
from ..keydict import KeySet, KeyDict
from ..small_scripts import mag
from ..repr_conventions import _repr


class NomialData(object):
    """Object for holding cs, exps, and other basic 'nomial' properties.

    cs: array (coefficient of each monomial term)
    exps: tuple of {VarKey: float} (exponents of each monomial term)
    varlocs: {VarKey: list} (terms each variable appears in)
    units: pint.UnitsContainer
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, exps=None, cs=None, simplify=True):
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
        self._varkeys, self._values = None, None
        if hasattr(self.cs, "units"):
            self.units = Quantity(1, self.cs.units)  #pylint: disable=no-member
        else:
            self.units = None

        self._hashvalue = None

    __repr__ = _repr

    def __hash__(self):
        if self._hashvalue is None:
            # confirm lengths before calling zip
            assert len(self.exps) == len(self.cs)
            # identical pint.Quantity object instances have different hashes
            self._hashvalue = hash(tuple(list(zip(self.exps, mag(self.cs))) +
                                         [str(self.units)]))
        return self._hashvalue

    @property
    def varkeys(self):
        "The NomialData's varkeys, created when necessary for a substitution."
        if not self._varkeys:
            self._varkeys = KeySet(self.varlocs)
        return self._varkeys

    @property
    def values(self):
        "The NomialData's values, created when necessary."
        if not self._values:
            self._values = KeyDict({k: k.descr["value"] for k in self.varlocs
                                    if "value" in k.descr})
        return self._values

    def init_from_nomials(self, nomials):
        """Way to initialize from nomials. Calls __init__.
        Used by subclass __init__ methods.
        """
        exps = functools_reduce(add, (tuple(s.exps) for s in nomials))
        cs = np.hstack((mag(s.cs) for s in nomials))
        # nomials are already simplified, so simplify=False
        NomialData.__init__(self, exps, cs, simplify=False)
        self.units = tuple(s.units for s in nomials)

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
        varset = self.varkeys[var]
        if len(varset) == 0:
            return NomialData(exps=[{}], cs=[0], simplify=False)
        elif len(varset) > 1:
            raise ValueError("multiple variables %s found for key %s"
                             % (list(varset), var))
        else:
            var, = varset
        exps, cs = [], []
        # var.units may be str if units disabled
        if not var.units or isinstance(var.units, str):
            units = 1
        else:
            if hasattr(self.cs, "units"):
                cs_units = Quantity(1, self.cs.units)
            else:
                cs_units = 1
            units = cs_units/var.units
        for i, exp in enumerate(self.exps):
            exp = HashVector(exp)   # copy -- exp is mutated below
            e = exp.get(var, 0)
            if var in exp:
                exp[var] -= 1
            exps.append(exp)
            cs.append(e*mag(self.cs)[i])
        # don't simplify to keep length same as self
        return NomialData(exps=exps, cs=cs*units, simplify=False)

    def __eq__(self, other):
        """Equality test"""
        if not all(hasattr(other, a) for a in ("exps", "cs")):
            return NotImplemented
        return self.exps == other.exps and all(self.cs == other.cs)


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
    if isinstance(cs, Quantity):
        units = cs.units
        cs = cs.magnitude
    elif isinstance(cs[0], Quantity):
        units = cs[0].units
        if len(cs) == 1:
            cs = [cs[0].magnitude]
        else:
            cs = [c.to(units).magnitude for c in cs]
    else:
        units = None
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
    cs_ = Quantity(cs_, units) if units else np.array(cs_, dtype='float')

    if not return_map:
        return exps_, cs_
    else:
        mmap = [HashVector() for c in cs_]
        for i, item in enumerate(matches.items()):
            exp, c = item
            for j in expmap[exp]:
                mmap[i][j] = mag(expmap[exp][j]/c)
        return exps_, cs_, mmap
