# -*- coding: utf-8 -*-
"""Module for creating singlet Monomials and vectors of singlet Monomials.

    Examples
    --------
    >>> pi = gpkit.Variable("\\pi", "half of the circle constant")

"""

from gpkit import PosyArray
from gpkit import Monomial


def Variable(name, *descr):
    """A described singlet Monomial.

    Parameters
    ----------
    name : str
        The variable's name; can be any string.
    descr : list
        May be either [units, label] or [label]

    Returns
    -------
    Monomial of named variable
    """
    descr = _format_description(descr)
    return Monomial(name, var_descrs={name: descr})


def VectorVariable(length, name, *descr):
    """A described vector of singlet Monomials.

    Parameters
    ----------
    length : int
        Length of vector.
    name : str
        The variable's name; can be any string.
    descr : list
        May be either [units, label] or [label]

    Returns
    -------
    PosyArray of Monomials, each containing a variable with the name '$V_{i}',
    where V is the vector's name and i is the variable's index.
    """
    descr = _format_description(descr)
    m = PosyArray([Monomial("%s_{%i}" % (name, i)) for i in xrange(length)])
    for el in m:
        el.var_descrs[el.exp.keys()[0]] = descr
    return m


def _format_description(descr):
    """Parses description lists

    Parameters
    ----------
    descr : list
        May be either [units, label] or [label]

    Returns
    -------
    units : str
    label : str
    """
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
