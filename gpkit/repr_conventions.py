"Repository for representation standards"
import sys
from .small_classes import Quantity

try:
    sys.stdout.write(u"\u200b")
    DEFAULT_UNIT_PRINTING = [":P~"]
except UnicodeEncodeError:
    DEFAULT_UNIT_PRINTING = [":~"]


def _repr(self):
    "Returns namespaced string."
    return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))


def _str(self):
    "Returns default string."
    return self.str_without()


def _repr_latex_(self):
    "Returns default latex for automatic iPython Notebook rendering."
    return "$$"+self.latex()+"$$"


def unitstr(units, into="%s", options=None, dimless=""):
    "Returns the string corresponding to an object's units."
    options = options or DEFAULT_UNIT_PRINTING[0]
    if hasattr(units, "units") and isinstance(units.units, Quantity):
        units = units.units
    if not isinstance(units, Quantity):
        return dimless
    rawstr = (u"{%s}" % options).format(units.units)
    if str(units.units) == "count":
        rawstr = u"count"  # TODO: remove when pint issue #356 is resolved
    units = rawstr.replace(" ", "").replace("dimensionless", dimless)
    return into % units or dimless
