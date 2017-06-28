"""Defines the VarKey class"""
from .small_classes import Strings, Quantity
from .small_scripts import unitstr, veckeyed


class Count(object):
    "Like python 2's itertools.count, for Python 3 compatibility."

    def __init__(self):
        self.count = -1

    def next(self):
        "Increment self.count and return it"
        self.count += 1
        return self.count


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
    eq_ignores = frozenset(["units", "value"])
    # ignore value in ==. Also skip units, since pints is weird and the unitstr
    #    will be compared anyway

    def __init__(self, name=None, **kwargs):
        self.descr = kwargs
        # Python arg handling guarantees 'name' won't appear in kwargs
        if isinstance(name, VarKey):
            self.descr.update(name.descr)
        else:
            if name is None:
                name = "\\fbox{%s}" % VarKey.new_unnamed_id()
            self.descr["name"] = str(name)

            # update in case user has disabled units
            from . import units as ureg
            if ureg and "units" in self.descr:
                units = self.descr["units"]
                if isinstance(units, Strings):
                    if units == "-":
                        del self.descr["units"]  # dimensionless
                    else:
                        self.descr["units"] = Quantity(1.0, units)
                elif isinstance(units, Quantity):
                    self.descr["units"] = units
                else:
                    raise ValueError("units must be either a string"
                                     " or a Quantity from gpkit.units.")

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
        self._hashvalue = hash(selfstr)
        self.key = self
        self.keys = set([self.name, selfstr,
                         self.str_without("models")])

        if "idx" in self.descr:
            self.veckey = veckeyed(self)
            self.keys.add(self.veckey)

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
                    substring = ", ".join(substring)
                string += "_%s" % (substring,)
        return string

    def __getattr__(self, attr):
        return self.descr.get(attr, None)

    def latex_unitstr(self):
        "Returns latex unitstr"
        us = unitstr(self.units, r"~\mathrm{%s}", "L~")
        utf = us.replace("frac", "tfrac").replace(r"\cdot", r"\cdot ")
        return utf if utf != r"~\mathrm{-}" else ""

    def latex(self, excluded=None):
        "Returns latex representation."
        if excluded is None:
            excluded = []
        string = self.name
        for subscript in self.subscripts:
            if subscript in self.descr and subscript not in excluded:
                substring = self.descr[subscript]
                if subscript == "models":
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
        if self.descr["name"] != other.descr["name"]:
            return False
        keyset = set(self.descr.keys())
        keyset = keyset.symmetric_difference(other.descr.keys())
        if keyset - self.eq_ignores:
            return False
        for key in self.descr:
            if key not in self.eq_ignores:
                if self.descr[key] != other.descr[key]:
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)
