import numpy as np

from collections import defaultdict
from collections import Iterable

from .small_classes import HashVector
from .small_classes import Strings, Numbers

from . import units as ureg
Quantity = ureg.Quantity

def isequal(a, b):
    if isinstance(a, Iterable) and not isinstance(a, Strings+(list, dict)):
        for i, a_i in enumerate(a):
            if isinstance(a_i, Iterable) and not isinstance(a_i, Strings+(list, dict)):
                if not isequal(a_i, b[i]):
                    return False
    elif a != b:
        return False
    return True

def link(gps, varids):
    if not isinstance(gps, Iterable):
        gps = [gps]
    if not isinstance(varids, Iterable):
        varids = [varids]

    def getvarkey(var):
        if isinstance(var, str):
            return gps[0].varkeys[var]
        else:
            # assume is VarKey or Monomial
            return var

    def getvarstr(var):
        if isinstance(var, str):
            return var
        else:
            # assume is VarKey or Monomial
            if hasattr(var, "_cmpstr"):
                return var._cmpstr
            else:
                return var.exp.keys()[0]._cmpstr

    if isinstance(varids, dict):
        subs = {getvarstr(k): getvarkey(v) for k, v in varids.items()}
    else:
        subs = {getvarstr(v): getvarkey(v) for v in varids}

    for gp in gps:
        gp.sub(subs)

    gppile = gps[0]
    for gp in gps[1:]:
        gppile += gp
    return gppile

def mag(c):
    "Return magnitude of a Number or Quantity"
    if isinstance(c, Quantity):
        return c.magnitude
    else:
        return c


def unitstr(units, into="%s", options="~"):
    if hasattr(units, "descr"):
        if isinstance(units.descr, dict):
            units = units.descr.get("units", "-")
    if units and not isinstance(units, Strings):
        try:
            rawstr = ("{:%s}" % options).format(units)
        except:
            rawstr = "1.0 " + str(units.units)
        units = "".join(rawstr.replace("dimensionless", "-").split()[1:])
    if units:
        return into % units
    else:
        return ""


def is_sweepvar(sub):
    "Determines if a given substitution indicates a sweep."
    try:
        if sub[0] == "sweep":
            if isinstance(sub[1], Iterable):
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

def locate_vars(exps):
    "From exponents form a dictionary of which monomials each variable is in."
    varlocs = defaultdict(list)
    varkeys = defaultdict(list)
    for i, exp in enumerate(exps):
        for var in exp:
            varlocs[var].append(i)
            if var not in varkeys[var.name]:
                varkeys[var.name].append(var)

    for name, varl in varkeys.items():
        if "length" in varl[0].descr:
            # vector var
            newlist = [None]*varl[0].descr["length"]
            for var in varl:
                newlist[var.descr["idx"]] = var
            varkeys[name] = newlist
        else:
            if len(varl) == 1:
                varkeys[name] = varl[0]
            else:
                varkeys[name] = []
                for var in varl:
                    if "model" in var.descr:
                        varkeys[name+"_%s"%var.descr["model"]] = var
                    else:
                        varkeys[name].append(var)
                if len(varkeys[name]) == 1:
                    varkeys[name] = varkeys[name][0]
                elif len(varkeys[name]) == 0:
                    del varkeys[name]

    return dict(varlocs), dict(varkeys)


def sort_and_simplify(exps, cs):
    "Reduces the number of monomials, and casts them to a sorted form."
    matches = defaultdict(float)
    for i, exp in enumerate(exps):
        exp = HashVector({var: x for (var, x) in exp.items() if x != 0})
        matches[exp] += cs[i]
    cs_ = matches.values()
    if isinstance(cs_[0], Quantity):
        units = cs_[0]/cs_[0].magnitude
        cs_ = [c.to(units).magnitude for c in cs_] * units
    else:
        cs_ = np.array(cs_, dtype='float')
    return tuple(matches.keys()), cs_


def results_table(data, title, senss=False):
    strs = ["              | " + title]
    for var, val in sorted(data.items(), key=lambda x: str(x[0])):
        vector = True
        if not isinstance(val, Iterable):
            val = [val]
            vector = False
        elif not val.shape:
                val = val.flatten()
                vector = False
        label = var.descr.get('label', '')
        if senss:
            units = "-"
            minval = 1e-2
        else:
            units = unitstr(var)
            minval = 0
        if abs(max(val)) >= minval:
            for i, v in enumerate(val):
                strs += ["%13s" % (str(var) + (str(i) if vector else "")) +
                         " : %-8.3g " % v +
                         "[%s] %s" % (units, label)]
    strs += ["              |"]
    return "\n".join(strs)


def flatten(ible, classes):
    """Flatten an iterable that contains other iterables

    Parameters
    ----------
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
