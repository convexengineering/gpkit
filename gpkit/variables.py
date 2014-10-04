from gpkit import PosyArray
from gpkit import Monomial


def Variable(name, *descr):
    descr = _format_description(descr)
    return Monomial(name, var_descrs={name: descr})


def VectorVariable(length, name, *descr):
    descr = _format_description(descr)
    m = PosyArray([Monomial("%s_{%i}" % (name, i)) for i in xrange(length)])
    for el in m:
        el.var_descrs[el.exp.keys()[0]] = descr
    return m


def _format_description(descr):
    label = None
    if len(descr) == 1:
            if isinstance(descr[0], str):
                units = None
                label = descr[0]
    elif len(descr) == 2:
        units = descr[0]
        if units in ("[-]", "", "-", "[]"):
            units = None
        label = descr[1]
    else:
        raise TypeError("variable descriptions should consist"
                        " of at most two parts: units and a description")
    if not isinstance(label, str):
        raise TypeError("variable labels should be strings.")

    return units, label