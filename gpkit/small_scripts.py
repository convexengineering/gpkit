import numpy as np

from collections import defaultdict
from collections import Iterable

from .small_classes import HashVector
from .small_classes import Strings, Quantity


def diff(p, vk):
    exps, cs = [], []
    from . import units as ureg
    units = vk.descr.get("units", 1) if ureg else 1
    for i, exp in enumerate(p.exps):
        exp = HashVector(exp)
        if vk in exp:
            e = exp[vk]
            if e == 1:
                del exp[vk]
            else:
                exp[vk] -= 1
            exps.append(exp)
            cs.append(e*p.cs[i]/units)
    return exps, cs


def mono_approx(p, x0):
    if not x0:
        for i, exp in enumerate(p.exps):
            if exp == {}:
                return p.cs[i], {}
    exp = HashVector()
    p0 = p.sub(x0).c
    m0 = 1
    from . import units as ureg
    for vk in p.varlocs:
        units = vk.descr.get("units", 1) if ureg else 1
        e = mag(x0[vk]*units * p.diff(vk).sub(x0, require_positive=False).c / p0)
        exp[vk] = e
        m0 *= (x0[vk]*units)**e
    return p0/m0, exp


def isequal(a, b):
    if (isinstance(a, Iterable)
       and not isinstance(a, Strings+(tuple, list, dict))):
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


def locate_vars(exps):
    "From exponents form a dictionary of which monomials each variable is in."
    varlocs = defaultdict(list)
    varkeys = defaultdict(set)
    for i, exp in enumerate(exps):
        for var in exp:
            varlocs[var].append(i)
            varkeys[var.name].add(var)

    varkeys_ = dict(varkeys)
    for name, varl in varkeys_.items():
        for vk in varl:
            descr = vk.descr
            break
        if "shape" in descr:
            # vector var
            newlist = np.zeros(descr["shape"], dtype="object")
            for var in varl:
                newlist[var.descr["idx"]] = var
            varkeys[name] = newlist
        else:
            if len(varl) == 1:
                varkeys[name] = varl.pop()
            else:
                varkeys[name] = []
                for var in varl:
                    if "model" in var.descr:
                        varkeys[name+"_%s" % var.descr["model"]] = var
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

    if matches[HashVector({})] == 0 and len(matches) > 1:
        del matches[HashVector({})]
    for exp, c in matches.items():
        if c == 0 and len(matches) > 1:
            del matches[exp]

    cs_ = list(matches.values())
    if isinstance(cs_[0], Quantity):
        units = cs_[0]/cs_[0].magnitude
        cs_ = [c.to(units).magnitude for c in cs_] * units
    else:
        cs_ = np.array(cs_, dtype='float')
    return tuple(matches.keys()), cs_


def results_table(data, title, minval=0, printunits=True, fixedcols=True,
                  varfmt="%s : ", valfmt="%-.4g ", vecfmt="%-8.3g"):
    """
    Pretty string representation of a dict of VarKeys
    Iterable values are handled specially (partial printing)

    Arguments
    ---------
    data: dict whose keys are VarKey's
        data to represent in table
    title: string
    minval: float
        skip values with all(abs(value)) < minval
    printunits: bool
    fixedcols: bool
        if True, print rhs (val, units, label) in fixed-width cols
    varfmt: string
        format for variable names
    valfmt: string
        format for scalar values
    vecfmt: string
        format for vector values
    """
    lines = []
    decorated = [(bool(v.shape) if isinstance(v, Iterable) else False,
                 (varfmt % k),
                 i, k, v) for i, (k, v) in enumerate(data.items())
                 if (np.max(abs(v)) >= minval) or (np.any(np.isnan(v)))]
                #                               allows nans to be printed 
    decorated.sort()
    for isvector, varstr, _, var, val in decorated:
        label = var.descr.get('label', '')
        units = unitstr(var, into=" [%s] ", dimless="") if printunits else ""
        if isvector:
            vals = [vecfmt % v for v in val[:4]]
            ellipsis = " ..." if len(val) > 4 else ""
            valstr = "[ %s%s ] " % ("  ".join(vals), ellipsis)
        else:
            valstr = valfmt % val
        lines.append([varstr, valstr, units, label])
    if lines:
        maxlens = np.max([map(len, line) for line in lines], axis=0)
        if not fixedcols:
            maxlens = [maxlens[0], 0, 0, 0]
        dirs = ['>', '<', '<', '<']
        assert len(dirs) == len(maxlens)  # check lengths before using zip
        fmts = ['{0:%s%s}' % (direc, L) for direc, L in zip(dirs, maxlens)]
    lines = [[fmt.format(s) for fmt, s in zip(fmts, line)]
             for line in lines]
    lines = [title] + ["-"*len(title)] + [''.join(l) for l in lines] + [""]
    return "\n".join(lines)


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
