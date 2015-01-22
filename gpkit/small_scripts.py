import numpy as np

from collections import defaultdict
from collections import Iterable

from .small_classes import HashVector

from . import units as ureg
Quantity = ureg.Quantity


def mag(c):
    "Return magnitude of a Number or Quantity"
    if isinstance(c, Quantity):
        return c.magnitude
    else:
        return c


def unitstr(v, into="%s", options="~"):
    '''
    Creates string of appropriate units

    Parameters
    ----------
    v :         Variable Name
                e.g.: C_D
                Type: VarKey

    into :      Output format (Default value: "%s")
                e.g.: "%s"
                Type: string

    options :   Options (Default value: "~")
                Type: string 

    Returns
    -------
    <string> :  String of unit name OR empty string
                e.g.: 'kg/m^3' 

    '''
    units = None
    if isinstance(v, Quantity):
        units = v
    elif hasattr(v, "descr"):
        if isinstance(v.descr, dict):
            units = v.descr.get("units", "-")
    if isinstance(units, Quantity):
        rawstr = ("{:%s}" % options).format(units)
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
    "Converts a number in scientific notation to a more latex-friendly string"
    cstr = "%.4g" % c
    if 'e' in cstr:
        idx = cstr.index('e')
        cstr = "%s \\times 10^{%i}" % (cstr[:idx], int(cstr[idx+1:]))
    return cstr


def locate_vars(exps):
    "From exponents form a dictionary of which monomials each variable is in."
    var_locs = defaultdict(list)
    for i, exp in enumerate(exps):
        for var in exp:
            var_locs[var].append(i)
    return var_locs


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
    """
    Creates results table

    Parameters
    ----------
    data :      dictionary containing variable-value pairs
            
    title :     Title of results table (e.g 'Free variables (mean)')

    senss :     Senstitivity table flag
                Should values be treated as sensitivities in this table
                If below 1e-2, don't print sensitivities
                Sensitivities have no units

    Returns
    -------
    <string> :  Results Table
                
    """
    strs = ["                    | " + title]
    for var, table in sorted(data.items(), key=lambda x: str(x[0])):
        try:
            val = table.mean()
        except AttributeError:
            val = table
        label = var.descr.get('label', '')
        if senss:
            units = "-"
            minval = 1e-2
        else:
            units = unitstr(var)
            minval = 0
        if abs(val) >= minval:
            strs += ["%19s" % var +
                     " : %-8.3g " % val +
                     "[%s] %s" % (units, label)]
    strs += ["                    |"]
    return "\n".join(strs)


def flatten(ible, classes):
    """Flatten an iterable that contains other iterables

    Parameters
    ----------
    ible :      Iterable
                Top-level container

    classes :   classes that are allowed

    Returns
    -------
    out :       list
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
