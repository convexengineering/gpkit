"""Defines the VarKey class"""
from .small_classes import HashVector, Count, qty
from .repr_conventions import unitstr


class VarKey(object):  # pylint:disable=too-many-instance-attributes
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
    subscripts = ("models", "idx")

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
        self.cleanstr = self.str_without(["modelnums"])
        self.eqstr = self.cleanstr + str(self.modelnums) + self.unitrepr
        self._hashvalue = hash(self.eqstr)
        self.keys = set((self.name, self.cleanstr))

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

    def str_without(self, excluded=None):
        "Returns string without certain fields (such as 'models')."
        if excluded is None:
            excluded = []
        string = self.name
        for subscript in self.subscripts:
            if self.descr.get(subscript) and subscript not in excluded:
                substring = self.descr[subscript]
                if subscript == "models":
                    if self.modelnums and "modelnums" not in excluded:
                        substring = ["%s.%s" % (ss, mn) if mn > 0 else ss
                                     for ss, mn
                                     in zip(substring, self.modelnums)]
                    substring = "/".join(substring)
                string += "_%s" % (substring,)
        return string

    def __getattr__(self, attr):
        return self.descr.get(attr, None)

    unitstr = unitstr

    def latex_unitstr(self):
        "Returns latex unitstr"
        us = self.unitstr(r"~\mathrm{%s}", ":L~")
        utf = us.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
        return utf if utf != r"~\mathrm{-}" else ""

    @property
    def naming(self):
        "Returns this varkey's naming tuple"
        # TODO: store naming (as special object?) instead of models/modelnums
        return (tuple(self.descr["models"]),
                tuple(self.descr["modelnums"]))

    def latex(self, excluded=None):
        "Returns latex representation."
        if excluded is None:
            excluded = []
        string = self.name
        for subscript in self.subscripts:
            if subscript in self.descr and subscript not in excluded:
                substring = self.descr[subscript]
                if subscript == "models":
                    if self.modelnums and "modelnums" not in excluded:
                        substring = ["%s.%s" % (ss, mn) if mn > 0 else ss
                                     for ss, mn
                                     in zip(substring, self.modelnums)]
                    substring = ", ".join(substring)
                string = "{%s}_{%s}" % (string, substring)
                if subscript == "idx":
                    if len(self.descr["idx"]) == 1:
                        # drop the comma for 1-d vectors
                        string = string[:-3]+string[-2:]
        if self.shape and not self.idx:
            string = "\\vec{%s}" % string  # add vector arrow for veckeys
        return string

    def _repr_latex_(self):
        return "$$"+self.latex()+"$$"

    def __hash__(self):
        return self._hashvalue

    def __eq__(self, other):
        if not hasattr(other, "descr"):
            return False
        return self.eqstr == other.eqstr

    def __ne__(self, other):
        return not self.__eq__(other)

from .nomials import NomialMap  # pylint: disable=wrong-import-position
