"""Assorted helper methods"""
from collections import Iterable
from .small_classes import Strings, Quantity


def isequal(a, b):
    if (isinstance(a, Iterable) and
            not isinstance(a, Strings+(tuple, list, dict))):
        for i, a_i in enumerate(a):
            if not isequal(a_i, b[i]):
                return False
    elif a != b:
        return False
    return True


def mag(c):
    "Return magnitude of a Number or Quantity"
    if isinstance(c, Quantity):
        return c.magnitude
    else:
        return c


def unitstr(units, into="%s", options="~", dimless='-'):
    if hasattr(units, "descr"):
        if isinstance(units.descr, dict):
            units = units.descr.get("units", dimless)
    if units and not isinstance(units, Strings):
        try:
            rawstr = ("{:%s}" % options).format(units)
        except:
            rawstr = "1.0 " + str(units.units)
        units = "".join(rawstr.replace("dimensionless", dimless).split()[1:])
    if units:
        return into % units
    else:
        return ""


def is_sweepvar(sub):
    "Determines if a given substitution indicates a sweep."
    try:
        if sub[0] == "sweep":
            if isinstance(sub[1], Iterable) or hasattr(sub[1], "__call__"):
                return True
    except:
        return False


def invalid_types_for_oper(oper, a, b):
    "Raises TypeError for unsupported operations."
    typea = a.__class__.__name__
    typeb = b.__class__.__name__
    raise TypeError("unsupported operand types"
                    " for %s: '%s' and '%s'" % (oper, typea, typeb))


def latex_num(c):
    cstr = "%.4g" % c
    if 'e' in cstr:
        idx = cstr.index('e')
        cstr = "%s \\times 10^{%i}" % (cstr[:idx], int(cstr[idx+1:]))
    return cstr


def flatten(ible, classes):
    """Flatten an iterable that contains other iterables

    Arguments
    ---------
    l : Iterable
        Top-level container

    Returns
    -------
    out : list
        List of all objects found in the nested iterables

    Raises
    ------
    TypeError
        If an object is found whose class was not in classes
    """
    out = []
    for el in ible:
        if isinstance(el, classes):
            out.append(el)
        elif isinstance(el, Iterable):
            for elel in flatten(el, classes):
                out.append(elel)
        else:
            raise TypeError("Iterable %s contains element '%s'"
                            " of invalid class %s." % (ible, el, el.__class__))
    return out
