"""Defines the VarKey class"""
import numpy as np

from .posyarray import PosyArray
from .small_scripts import mag
from .small_scripts import isequal
from .small_classes import Strings, Quantity
from .small_classes import Counter


class VarKey(object):
    """An object to correspond to each 'variable name'.

    Arguments
    ---------
    name : str, VarKey, or Monomial
        Name of this Variable, or object to derive this Variable from.

    **kwargs :
        Any additional attributes, which become the descr attribute (a dict).

    Returns
    -------
    VarKey with the given name and descr.
    """
    new_unnamed_id = Counter()

    def __init__(self, name=None, **kwargs):
        self.descr = kwargs
        # Python arg handling guarantees 'name' won't appear in kwargs
        if isinstance(name, VarKey):
            self.descr.update(name.descr)
        elif hasattr(name, "c") and hasattr(name, "exp"):
            if mag(name.c) == 1 and len(name.exp) == 1:
                var = list(name.exp)[0]
                self.descr.update(var.descr)
            else:
                raise TypeError("variables can only be formed from monomials"
                                " with a c of 1 and a single variable")
        else:
            if name is None:
                name = "\\fbox{%s}" % VarKey.new_unnamed_id()
            self.descr["name"] = str(name)

        from . import units as ureg  # update in case user has disabled units

        if "value" in self.descr:
            value = self.descr["value"]
            if isinstance(value, Quantity):
                self.descr["value"] = value.magnitude
                self.descr["units"] = value/value.magnitude
        if ureg and "units" in self.descr:
            units = self.descr["units"]
            if isinstance(units, Strings):
                units = units.replace("-", "dimensionless")
                self.descr["units"] = 1.0*ureg.parse_expression(units)
            elif isinstance(units, Quantity):
                self.descr["units"] = units/units.magnitude
            else:
                raise ValueError("units must be either a string"
                                 " or a Quantity from gpkit.units.")
        self._hashvalue = hash(self._cmpstr)

    @property
    def name(self):
        """name of this VarKey"""
        return self.descr['name']

    @property
    def units(self):
        """units of this VarKey"""
        return self.descr.get("units", None)

    def __repr__(self):
        s = self.name
        for subscript in ["model", "idx"]:
            if subscript in self.descr:
                s = "%s_%s" % (s, self.descr[subscript])
        return s

    def _latex(self, unused=None):
        s = self.name
        for subscript in ["idx"]:  # +"model"?
            if subscript in self.descr:
                s = "{%s}_{%s}" % (s, self.descr[subscript])
        return s

    @property
    def _cmpstr(self):
        s = self.name
        for subscript in ["idx"]:
            if subscript in self.descr:
                s = "%s_%s" % (s, self.descr[subscript])
        return s

    def __hash__(self):
        return self._hashvalue

    def __eq__(self, other):
        if isinstance(other, VarKey):
            if set(self.descr.keys()) != set(other.descr.keys()):
                return False
            for key in self.descr:
                if key == "units":
                    try:
                        if not self.descr["units"] == other.descr["units"]:
                            d = self.descr["units"]/other.descr["units"]
                            if str(d.units) != "dimensionless":
                                if not abs(mag(d)-1.0) <= 1e-7:
                                    return False
                    except:
                        return False
                else:
                    if not isequal(self.descr[key], other.descr[key]):
                        return False
            return True
        elif isinstance(other, Strings):
            return self._cmpstr == other
        elif isinstance(other, PosyArray):
            it = np.nditer(other, flags=['multi_index', 'refs_ok'])
            while not it.finished:
                i = it.multi_index
                it.iternext()
                p = other[i]
                v = VarKey(list(p.exp)[0])
                if v.descr.pop("idx", None) != i:
                    return False
                if v != self:
                    return False
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
