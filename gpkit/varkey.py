"""Defines the VarKey class"""
from .small_classes import Strings, Quantity, HashVector, Count
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

    def __init__(self, name=None, **kwargs):
        from . import units as ureg  # update in case user disabled units
        # NOTE: Python arg handling guarantees 'name' won't appear in kwargs
        self.descr = kwargs
        if isinstance(name, VarKey):
            self.descr.update(name.descr)
        else:
            if name is None:
                name = "\\fbox{%s}" % VarKey.new_unnamed_id()
            self.descr["name"] = str(name)
            if "units" in self.descr:
                units = self.descr["units"]
                if ureg:
                    if isinstance(units, Strings):
                        if units in ["", "-"]:  # dimensionless
                            del self.descr["units"]
                        else:
                            self.descr["units"] = Quantity(1.0, units)
                    elif not isinstance(units, Quantity):
                        raise ValueError("units must be either a string"
                                         " or a Quantity from gpkit.units.")
                else:  # units are disabled
                    del self.descr["units"]
                    self.descr["unitslabel"] = units

            if "value" in self.descr:
                value = self.descr["value"]
                if isinstance(value, Quantity):
                    if "units" in self.descr:
                        # convert to explicitly given units, if any
                        value = value.to(self.descr["units"])
                    else:
                        self.descr["units"] = Quantity(1.0, value.units)
                    self.descr["value"] = value.magnitude

            self.descr["unitrepr"] = repr(self.units)

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
