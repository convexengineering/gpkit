# -*- coding: utf-8 -*-
"""Various helper functions for creating described Monomials and PosyArrays

    Example
    -------
    >>> x = gpkit.mon('x', 2, 'm')
    >>> y = gpkit.vecmon(3, 'y', [1, 2, 3], 'm')

"""

from collections import Iterable

from .small_classes import Numbers, Strings
from . import units as ureg
Quantity = ureg.Quantity

from .nomials import *

from small_scripts import is_sweepvar


def mon(*args, **descr):
    """A described singlet Monomial.

    Parameters
    ----------
    *args : list
        may contain "name" (Strings)
                    "value" (Numbers + Quantity) or (Iterable) for a sweep
                    "units" (Strings + Quantity)
             and/or "label" (Strings)
    **descr : dict
        Variable description

    Returns
    -------
    Monomials containing a variable with the name '$name',
    where $name is the vector's name and i is the variable's index.
    """
    for arg in args:
        if isinstance(arg, Strings) and "name" not in descr:
            descr["name"] = arg
        elif isinstance(arg, Numbers + (Quantity,)) and "value" not in descr:
            descr["value"] = arg
        elif (isinstance(arg, Iterable) and not isinstance(arg, Strings)
              and "value" not in descr):
            if is_sweepvar(arg):
                descr["value"] = arg
            else:
                descr["value"] = ("sweep", arg)
        elif isinstance(arg, Strings + (Quantity,)) and "units" not in descr:
            descr["units"] = arg
        elif isinstance(arg, Strings) and "label" not in descr:
            descr["label"] = arg
    return Monomial(**descr)


def vecmon(length, *args, **descr):
    """A described vector of singlet Monomials.

    Parameters
    ----------
    length : int
        Length of vector.
    *args : list
        may contain "name" (Strings)
                    "value" (Iterable)
                    "units" (Strings + Quantity)
             and/or "label" (Strings)
    **descr : dict
        Variable description

    Returns
    -------
    PosyArray of Monomials, each containing a variable with name '$name_{i}',
    where $name is the vector's name and i is the variable's index.
    """
    if "idx" in descr:
        raise KeyError("the description field 'idx' is reserved")

    descr["length"] = length

    for arg in args:
        if isinstance(arg, Strings) and "name" not in descr:
            descr["name"] = arg
        elif (isinstance(arg, Iterable) and not isinstance(arg, Strings)
              and "value" not in descr):
            descr["value"] = arg
        elif isinstance(arg, Strings + (Quantity,)) and "units" not in descr:
            descr["units"] = arg
        elif isinstance(arg, Strings) and "label" not in descr:
            descr["label"] = arg

    values = descr.pop("value", [])
    if values and len(values) != length:
        raise ValueError("vector length and values length must be the same.")

    vl = []
    for i in range(length):
        descr.update({"idx": i})
        if values:
            descr.update({"value": values[i]})
        vl.append(Monomial(**descr))
    vm = PosyArray(vl)
    vm.descr = dict(vm[0].exp.keys()[0].descr)
    vm.descr.pop("idx", None)
    return vm
