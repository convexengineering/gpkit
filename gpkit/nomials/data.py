"""Machinery for exps, cs, varlocs data -- common to nomials and programs"""
from collections import defaultdict
import numpy as np
from ..small_classes import HashVector
from ..keydict import KeySet
from .map import NomialMap
from ..repr_conventions import _repr
from ..varkey import VarKey


class NomialData(object):
    """Object for holding cs, exps, and other basic 'nomial' properties.

    cs: array (coefficient of each monomial term)
    exps: tuple of {VarKey: float} (exponents of each monomial term)
    varlocs: {VarKey: list} (terms each variable appears in)
    units: pint.UnitsContainer
    """
    # pylint: disable=too-many-instance-attributes
    _hashvalue = _varlocs = _exps = _cs = _varkeys = None

    def _reset(self):
        self._hashvalue = \
            self._varlocs = \
            self._exps = \
            self._cs = \
            self._varkeys = \
            self._values = None

    def __init__(self, hmap):
        self.hmap = hmap

        self.vks = set()
        for exp in self.hmap:
            self.vks.update(exp)
        self.units = self.hmap.units
        self.any_nonpositive_cs = any(c <= 0 for c in self.hmap.values())
        self._reset()

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
            self._cs = np.array(self.hmap.values())
            if self.hmap.units:
                self._cs = self._cs*self.hmap.units
        return self._cs

    __repr__ = _repr

    def __hash__(self):
        if self._hashvalue is None:
            self._hashvalue = hash(hash(self.hmap) + hash(str(self.hmap.units)))
        return self._hashvalue

    @property
    def varkeys(self):
        "The NomialData's varkeys, created when necessary for a substitution."
        if self._varkeys is None:
            self._varkeys = KeySet(self.vks)
        return self._varkeys

    @property
    def values(self):  # TODO: if it's none presume it stays that way?
        "The NomialData's values, created when necessary."
        return {k: k.descr["value"] for k in self.vks
                if "value" in k.descr}

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
        # pylint:disable=len-as-condition
        varset = self.varkeys[var]
        if len(varset) > 1:
            raise ValueError("multiple variables %s found for key %s"
                             % (list(varset), var))
        elif len(varset) == 0:
            hmap = NomialMap({HashVector(): 0})
            hmap.units = None
        else:
            var, = varset
            hmap = self.hmap.diff(var)
        return NomialData(hmap)

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
        return not self.__eq__(other)
