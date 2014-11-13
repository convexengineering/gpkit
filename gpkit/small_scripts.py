from collections import defaultdict
from collections import Iterable

from .small_classes import HashVector


def is_sweepvar(sub):
    "Determines if a given substitution indicates a sweep."
    try:
        assert sub[0] == "sweep"
        assert isinstance(sub[1], Iterable)
    except: return False


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
        cstr = "%s\\times 10^{%i}" % (cstr[:idx], int(cstr[idx+1:]))
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
    return tuple(matches.keys()), tuple(matches.values())


def print_results_table(data, title, senss=False):
    print("                    | " + title)
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
            units = var.descr.get('units', '-')
            minval = 0
        if abs(val) >= minval:
            print "%19s" % var, ": %-8.3g" % val, "[%s] %s" % (units, label)
    print("                    |")


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
            for elel in flatten_constr(el):
                out.append(elel)
        else:
            raise TypeError("Iterable %s contains element '%s'"
                            " of invalid class %s." % (ible, el, el.__class__))
    return out
