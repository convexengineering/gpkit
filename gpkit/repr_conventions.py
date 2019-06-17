"Repository for representation standards"
import sys
from .small_classes import Quantity

try:
    sys.stdout.write(u"\u200b")
    DEFAULT_UNIT_PRINTING = [":P~"]
except UnicodeEncodeError:
    DEFAULT_UNIT_PRINTING = [":~"]


def lineagestr(lineage, modelnums=True):
    "Returns properly formatted lineage string"
    lineage = getattr(lineage, "lineage", None) or lineage
    return "/".join(["%s.%i" % (name, num) if (num and modelnums) else name
                     for name, num in lineage]) if lineage else ""


def unitstr(units, into="%s", options=None, dimless=""):
    "Returns the string corresponding to an object's units."
    options = options or DEFAULT_UNIT_PRINTING[0]
    if hasattr(units, "units") and isinstance(units.units, Quantity):
        units = units.units
    if not isinstance(units, Quantity):
        return dimless
    rawstr = (u"{%s}" % options).format(units.units)
    units = rawstr.replace(" ", "").replace("dimensionless", dimless)
    return into % units or dimless


class GPkitObject(object):
    "This class combines various printing methods for easier adoption."
    lineagestr = lineagestr
    unitstr = unitstr

    def __repr__(self):
        "Returns namespaced string."
        return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))

    def __str__(self):
        "Returns default string."
        return self.str_without()  # pylint: disable=no-member

    def _repr_latex_(self):
        "Returns default latex for automatic iPython Notebook rendering."
        return "$$"+self.latex()+"$$"  # pylint: disable=no-member
