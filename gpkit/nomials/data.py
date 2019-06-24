"""Machinery for exps, cs, varlocs data -- common to nomials and programs"""
from __future__ import unicode_literals
from collections import defaultdict
import numpy as np
from ..keydict import KeySet
from ..repr_conventions import GPkitObject
from ..varkey import VarKey


class NomialData(GPkitObject):
    """Object for holding cs, exps, and other basic 'nomial' properties.

    cs: array (coefficient of each monomial term)
    exps: tuple of {VarKey: float} (exponents of each monomial term)
    varlocs: {VarKey: list} (terms each variable appears in)
    units: pint.UnitsContainer
    """
    # pylint: disable=too-many-instance-attributes
    _hashvalue = _varlocs = _exps = _cs = _varkeys = None

    def __init__(self, hmap):
        self.hmap = hmap
        self.vks = set()
        for exp in self.hmap:
            self.vks.update(exp)
        self.units = self.hmap.units
        self.any_nonpositive_cs = any(c <= 0 for c in self.hmap.values())

    def to(self, units):
        "Create new Signomial converted to new units"
        return self.__class__(self.hmap.to(units))  # pylint: disable=no-member

    @property
    def varlocs(self):
        "Create varlocs or return cached varlocs"
        if self._varlocs is None:
            self._varlocs = defaultdict(list)
            for i, exp in enumerate(self.exps):
                for var in exp:
                    self._varlocs[var].append(i)
        return self._varlocs

    @property
    def exps(self):
        "Create exps or return cached exps"
        if self._exps is None:
            self._exps = tuple(self.hmap.keys())
        return self._exps

    @property
    def cs(self):
        "Create cs or return cached cs"
        if self._cs is None:
            self._cs = np.array(list(self.hmap.values()))
            if self.hmap.units:
                self._cs = self._cs*self.hmap.units
        return self._cs

    def __hash__(self):
        return hash(self.hmap)

    @property
    def varkeys(self):
        "The NomialData's varkeys, created when necessary for a substitution."
        if self._varkeys is None:
            self._varkeys = KeySet(self.vks)
        return self._varkeys

    def varkeyvalues(self):
        "Returns the NomialData's keys' values"
        return {k: k.descr["value"] for k in self.vks if "value" in k.descr}

    def __eq__(self, other):
        "Equality test"
        if not hasattr(other, "hmap"):
            return NotImplemented
        if isinstance(other, VarKey):
            return False
        if self.hmap != other.hmap:
            return False
        if self.units != other.units:
            return False
        return True

    def __ne__(self, other):
        return not self == other
