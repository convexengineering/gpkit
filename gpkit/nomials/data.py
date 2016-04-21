"""Machinery for exps, cs, varlocs data -- common to nomials and programs"""
from collections import defaultdict
from functools import reduce as functools_reduce
from operator import add
import numpy as np
from ..small_classes import HashVector, Quantity
from ..keydict import KeySet, KeyDict
from ..small_scripts import mag
from ..nomial_map import NomialMap


class NomialData(object):
    """Object for holding cs, exps, and other basic 'nomial' properties.

    cs: array (coefficient of each monomial term)
    exps: tuple of {VarKey: float} (exponents of each monomial term)
    varlocs: {VarKey: list} (terms each variable appears in)
    units: pint.UnitsContainer
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, hmap=None):
        self.hmap = hmap

        self.vks = {}
        for exp in self.hmap:
            for vk in exp:
                if vk not in self.vks:
                    self.vks[vk] = None

        self._hashvalue = None
        self.units = self.hmap.units
        self.any_nonpositive_cs = any(c <= 0 for c in self.hmap.values())

    @property
    def varlocs(self):
        if not hasattr(self, "_varlocs"):
            self._varlocs = {}
            for i, exp in enumerate(self.exps):
                for var in exp:
                    if var not in self._varlocs:
                        self._varlocs[var] = []
                    self._varlocs[var].append(i)
        return self._varlocs

    @property
    def exps(self):
        if not hasattr(self, "_exps"):
            self._exps = tuple(self.hmap.keys())
        return self._exps

    @property
    def cs(self):
        if not hasattr(self, "_cs"):
            self._cs = np.array(self.hmap.values())
            if self.hmap.units:
                self._cs = self._cs*self.hmap.units
        return self._cs

    def __hash__(self):
        if not self._hashvalue:
            self._hashvalue = hash(self.hmap)
        return self._hashvalue

    @classmethod
    def fromnomials(cls, nomials):
        """Construct a NomialData from an iterable of Signomial objects"""
        nd = cls()  # use pass-through version of __init__
        nd.init_from_nomials(nomials)
        return nd

    @property
    def varkeys(self):
        "The NomialData's varkeys, created when necessary for a substitution."
        if not hasattr(self, "_varkeys"):
            self._varkeys = KeySet(self.vks)
        return self._varkeys

    @property
    def values(self):
        "The NomialData's values, created when necessary."
        if not hasattr(self, "_values"):
            self._values = KeyDict({k: k.descr["value"] for k in self.vks
                                    if "value" in k.descr})
        return self._values

    def init_from_nomials(self, nomials):
        """Way to initialize from nomials. Calls __init__.
        Used by subclass __init__ methods.
        """
        exps = functools_reduce(add, (tuple(s.exps) for s in nomials))
        cs = np.hstack((mag(s.cs) for s in nomials))
        # nomials are already simplified, so simplify=False
        self._exps = exps
        self._cs = cs
        hmap = NomialMap.from_exps_and_cs(exps, cs)
        NomialData.__init__(self, hmap=hmap)
        self.units = tuple(s.units for s in nomials)

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, hash(self))

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
        units = self.units/var.units if self.units else None
        if len(varset) > 1:
            raise ValueError("multiple variables %s found for key %s"
                             % (list(varset), var))
        elif len(varset) == 0:
            hmap = NomialMap({HashVector(): 0})
        else:
            var, = varset
            exps, cs = [], []
            # var.units may be str if units disabled
            csmag = mag(self.cs)
            for i, exp in enumerate(self.exps):
                exp = HashVector(exp)   # copy -- exp is mutated below
                e = exp.get(var, 0)
                if var in exp:
                    exp[var] -= 1
                exps.append(exp)
                cs.append(e*csmag[i])
            # don't simplify to keep length same as self
            hmap = NomialMap.from_exps_and_cs(exps, cs)
        hmap.set_units(units)
        return NomialData(hmap=hmap)

    def __eq__(self, other):
        """Equality test"""
        if not all(hasattr(other, a) for a in ("exps", "cs", "units")):
            return NotImplemented
        if self.hmap != other.hmap:
            return False
        if self.units != other.units:
            return False
        return True
