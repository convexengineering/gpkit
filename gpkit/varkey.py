"""Defines the VarKey class"""
from .small_classes import Strings, Quantity, HashVector, Count, qty
from .small_scripts import veckeyed
from .repr_conventions import unitstr


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
    new_unnamed_id = Count().next
    subscripts = ("models", "idx")
    @profile
    def __init__(self, name=None, **kwargs):
        # NOTE: Python arg handling guarantees 'name' won't appear in kwargs
        self.descr = kwargs
        if isinstance(name, VarKey):
            self.descr.update(name.descr)
        else:
            if name is None:
                name = "\\fbox{%s}" % VarKey.new_unnamed_id()
            self.descr["name"] = str(name)
            if "unitrepr" in self.descr:
                self.descr["units"] = self.descr["unitrepr"]
            self.descr["unitrepr"] = "None"
            if "units" in self.descr:
                units = self.descr["units"]
                if isinstance(units, Strings):
                    if units in ["", "-", "None"]:  # dimensionless
                        del self.descr["units"]
                        self.descr["unitrepr"] = "None"
                    else:
                        self.descr["units"] = qty(units)
                        self.descr["unitrepr"] = units
                else:
                    raise ValueError("units must be given as a string")

            if isinstance(self.value, Quantity):
                raise ValueError("values may not be united")

        selfstr = str(self)
        self.hashstr = selfstr + self.descr["unitrepr"]
        self._hashvalue = hash(self.hashstr)
        self.key = self
        self.keys = set([self.name, selfstr,
                         self.str_without(["modelnums"])])

        if "idx" in self.descr:
            self.veckey = veckeyed(self)
            self.keys.add(self.veckey)
            self.keys.add(self.str_without(["idx"]))
            self.keys.add(self.str_without(["idx", "modelnums"]))

        self.hmap = NomialMap({HashVector({self: 1}): 1.0})
        self.hmap.units = self.units

    def __repr__(self):
        return self.str_without()

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
        us = self.unitstr(r"~\mathrm{%s}", "L~")
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
        return self.hashstr == other.hashstr

    def __ne__(self, other):
        return not self.__eq__(other)

from .nomials import NomialMap  # pylint: disable=wrong-import-position
