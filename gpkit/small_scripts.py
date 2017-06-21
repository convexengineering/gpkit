"""Assorted helper methods"""
from collections import Iterable
from .small_classes import Strings, Quantity


def try_str_without(item, excluded):
    "Try to call item.str_without(excluded); fall back to str(item)"
    if hasattr(item, "str_without"):
        return item.str_without(excluded)
    else:
        return str(item)


def veckeyed(key):
    "Return a veckey version of a VarKey"
    vecdescr = dict(key.descr)
    for metadata in ["idx", "value"]:
        vecdescr.pop(metadata, None)
    return key.__class__(**vecdescr)


def mag(c):
    "Return magnitude of a Number or Quantity"
    return getattr(c, "magnitude", c)


def unitstr(units, into="%s", options="~", dimless=""):
    "Returns the unitstr of a given object."
    if hasattr(units, "descr") and hasattr(units.descr, "get"):
        units = units.descr.get("units", None)
    if hasattr(units, "units") and isinstance(units.units, Quantity):
        units = units.units
    if isinstance(units, Strings):
        return into % units or dimless
    elif isinstance(units, Quantity):
        rawstr = (u"{:%s}" % options).format(units.units)
        if str(units.units) == "count":
            # TODO remove this conditional when pint issue 356 is resolved
            rawstr = u"count"
        units = rawstr.replace(" ", "").replace("dimensionless", dimless)
        return into % units or dimless
    return dimless


def nomial_latex_helper(c, pos_vars, neg_vars):
    """Combines (varlatex, exponent) tuples,
    separated by positive vs negative exponent,
    into a single latex string"""
    # TODO this is awkward due to sensitivity_map, which needs a refactor
    pvarstrs = ['%s^{%.2g}' % (varl, x) if "%.2g" % x != "1" else varl
                for (varl, x) in pos_vars]
    nvarstrs = ['%s^{%.2g}' % (varl, -x)
                if "%.2g" % -x != "1" else varl
                for (varl, x) in neg_vars]
    pvarstrs.sort()
    nvarstrs.sort()
    pvarstr = ' '.join(pvarstrs)
    nvarstr = ' '.join(nvarstrs)
    c = mag(c)
    cstr = "%.2g" % c
    if pos_vars and (cstr == "1" or cstr == "-1"):
        cstr = cstr[:-1]
    else:
        cstr = latex_num(c)

    if not pos_vars and not neg_vars:
        mstr = "%s" % cstr
    elif pos_vars and not neg_vars:
        mstr = "%s%s" % (cstr, pvarstr)
    elif neg_vars and not pos_vars:
        mstr = "\\frac{%s}{%s}" % (cstr, nvarstr)
    elif pos_vars and neg_vars:
        mstr = "%s\\frac{%s}{%s}" % (cstr, pvarstr, nvarstr)

    return mstr


def is_sweepvar(sub):
    "Determines if a given substitution indicates a sweep."
    try:
        sweep, value = sub
        if sweep is "sweep" and (isinstance(value, Iterable) or
                                 hasattr(value, "__call__")):
            return True
    except (TypeError, ValueError):
        pass
    return False


def latex_num(c):
    "Returns latex string of numbers, potentially using exponential notation."
    cstr = "%.4g" % c
    if 'e' in cstr:
        idx = cstr.index('e')
        cstr = "%s \\times 10^{%i}" % (cstr[:idx], int(cstr[idx+1:]))
    return cstr
