"""Defines the VarKey class"""
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
    subscripts = ("lineage", "idx")

    def __init__(self, name=None, **kwargs):
        # NOTE: Python arg handling guarantees 'name' won't appear in kwargs
        if isinstance(name, VarKey):
            self.descr = name.descr
        else:
            self.descr = kwargs
            self.descr["name"] = str(name or "\\fbox{%s}" % VarKey.unique_id())
            unitrepr = self.unitrepr or self.units
            if unitrepr in ["", "-", None]:  # dimensionless
                self.descr["units"] = None
                self.descr["unitrepr"] = "-"
            else:
                self.descr["units"] = qty(unitrepr)
                self.descr["unitrepr"] = unitrepr

        self.key = self
        fullstr = self.str_without(["modelnums"])
        self.eqstr = fullstr + str(self.lineage) + self.unitrepr
        self._hashvalue = hash(self.eqstr)
        self.keys = set((self.name, fullstr))

        if "idx" in self.descr:
            if "veckey" not in self.descr:
                vecdescr = self.descr.copy()
                del vecdescr["idx"]
                self.veckey = VarKey(**vecdescr)
            self.keys.add(self.veckey)
            self.keys.add(self.str_without(["idx"]))
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
        string = self.name
        for subscript in self.subscripts:
            if subscript in self.descr and subscript not in excluded:
                substring = self.descr[subscript]
                if subscript == "lineage":
                    substring = self.lineagestr("modelnums" not in excluded)
                string += "_%s" % (substring,)
        return string

    def __getattr__(self, attr):
        return self.descr.get(attr, None)

    @property
    def models(self):
        "Returns a tuple of just the names of models in self.lineage"
        return list(zip(*self.lineage))[0]

    def latex_unitstr(self):
        "Returns latex unitstr"
        us = self.unitstr(r"~\mathrm{%s}", ":L~")
        utf = us.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
        return utf if utf != r"~\mathrm{-}" else ""

    def latex(self, excluded=()):
        "Returns latex representation."
        string = self.name
        if self.shape and not self.idx:
            string = "\\vec{%s}" % string  # add vector arrow for veckeys
        for subscript in self.subscripts:
            if subscript in self.descr and subscript not in excluded:
                substring = self.descr[subscript]
                if subscript == "lineage":
                    substring = self.lineagestr("modelnums" not in excluded)
                elif subscript == "idx" and len(self.idx) == 1:
                    substring = self.idx[0]  # drop the tuple comma in 1D
                string = "{%s}_{%s}" % (string, substring)
        return string

    def __hash__(self):
        return self._hashvalue

    def __eq__(self, other):
        if not hasattr(other, "descr"):
            return False
        return self.eqstr == other.eqstr

    def __ne__(self, other):
        return not self == other

from .nomials import NomialMap  # pylint: disable=wrong-import-position
