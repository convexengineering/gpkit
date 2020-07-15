"""Defines the VarKey class"""
from .small_classes import HashVector, Count, qty
from .repr_conventions import ReprMixin


class VarKey(ReprMixin):  # pylint:disable=too-many-instance-attributes
    """An object to correspond to each 'variable name'.

    Arguments
    ---------
    name : str, VarKey, or Monomial
        Name of this Variable, or object to derive this Variable from.

    **descr :
        Any additional attributes, which become the descr attribute (a dict).

    Returns
    -------
    VarKey with the given name and descr.
    """
    unique_id = Count().next
    vars_of_a_name = {}
    subscripts = ("lineage", "idx")

    def __init__(self, name=None, **descr):
        # NOTE: Python arg handling guarantees 'name' won't appear in descr
        self.descr = descr
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
        self.hashvalue = hash(self.eqstr)
        self.keys = set((self.name, fullstr))
        self.hmap = NomialMap({HashVector({self: 1}): 1.0})
        self.hmap.units = self.units

        if "idx" in self.descr:
            if "veckey" not in self.descr:
                vecdescr = self.descr.copy()
                del vecdescr["idx"]
                self.veckey = VarKey(**vecdescr)
            else:
                self.keys.add(self.veckey)
                self.keys.add(self.str_without(["idx", "modelnums"]))

    def __getstate__(self):
        "Stores varkey as its metadata dictionary, removing functions"
        state = self.descr.copy()
        for key, value in state.items():
            if getattr(value, "__call__", None):
                state[key] = "unpickleable function %s" % value
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

    __repr__ = str_without

    # pylint: disable=multiple-statements
    def __hash__(self): return self.hashvalue
    def __getattr__(self, attr): return self.descr.get(attr, None)

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

    def __eq__(self, other):
        if not hasattr(other, "descr"):
            return False
        return self.eqstr == other.eqstr

from .nomials import NomialMap  # pylint: disable=wrong-import-position
