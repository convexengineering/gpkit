"""Defines the VarKey class"""
from __future__ import unicode_literals
from .small_classes import HashVector, Count, qty
from .repr_conventions import GPkitObject


class VarKey(GPkitObject):  # pylint:disable=too-many-instance-attributes
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
    unique_id = Count().next
    vars_of_a_name = {}
    subscripts = ("lineage", "idx")

    def __init__(self, name=None, **kwargs):
        # NOTE: Python arg handling guarantees 'name' won't appear in kwargs
        self.descr = kwargs
        self.descr["name"] = name or "\\fbox{%s}" % VarKey.unique_id()
        unitrepr = self.unitrepr or self.units
        if unitrepr in ["", "-", None]:  # dimensionless
            self.descr["units"] = None
            self.descr["unitrepr"] = "-"
        else:
            self.descr["units"] = qty(unitrepr)
            self.descr["unitrepr"] = unitrepr

        self.key = self
        fullstr = self.str_without(["modelnums", "vec"])
        self.eqstr = fullstr + str(self.lineage) + self.unitrepr
        self._hashvalue = hash(self.eqstr)
        self.keys = set((self.name, fullstr))

        if "idx" in self.descr:
            if "veckey" not in self.descr:
                vecdescr = self.descr.copy()
                del vecdescr["idx"]
                self.veckey = VarKey(**vecdescr)
            else:
                self.keys.add(self.veckey)
                self.keys.add(self.str_without(["idx", "modelnums"]))

        self.hmap = NomialMap({HashVector({self: 1}): 1.0})
        self.hmap.units = self.units

    def __repr__(self):
        return self.str_without()

    def __getstate__(self):
        "Stores varkey as its metadata dictionary, removing functions"
        state = self.descr.copy()
        state.pop("units", None)  # not necessary, but saves space
        for key, value in state.items():
            if getattr(value, "__call__", None):
                state[key] = str(value)
        return state

    def __setstate__(self, state):
        "Restores varkey from its metadata dictionary"
        self.__init__(**state)

    def str_without(self, excluded=()):
        "Returns string without certain fields (such as 'lineage')."
        name = self.name
        if ("lineage" not in excluded and self.lineage
                and ("unnecessary lineage" not in excluded
                     or self.necessarylineage)):
            name = self.lineagestr("modelnums" not in excluded) + "." + name
        if "idx" not in excluded:
            if self.idx:
                name += "[%s]" % ",".join(map(str, self.idx))
            elif "vec" not in excluded and self.shape:
                name += "[:]"
        return name

    def __getattr__(self, attr):
        return self.descr.get(attr, None)

    @property
    def models(self):
        "Returns a tuple of just the names of models in self.lineage"
        return list(zip(*self.lineage))[0]

    def latex(self, excluded=()):
        "Returns latex representation."
        name = self.name
        if "vec" not in excluded and "idx" not in excluded and self.shape:
            name = "\\vec{%s}" % name
        if "idx" not in excluded and self.idx:
            name = "{%s}_{%s}" % (name, ",".join(map(str, self.idx)))
        if ("lineage" not in excluded and self.lineage
                and ("unnecessary lineage" not in excluded
                     or self.necessarylineage)):
            name = "{%s}_{%s}" % (name,
                                  self.lineagestr("modelnums" not in excluded))
        return name

    def __hash__(self):
        return self._hashvalue

    def __eq__(self, other):
        if not hasattr(other, "descr"):
            return False
        return self.eqstr == other.eqstr

    def __ne__(self, other):
        return not self == other

from .nomials import NomialMap  # pylint: disable=wrong-import-position
