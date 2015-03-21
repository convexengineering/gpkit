import numpy as np
from collections import Iterable

from .varkey import VarKey
from .nomials import Monomial
from .posyarray import PosyArray
from .small_classes import Strings, Numbers
from .small_scripts import is_sweepvar

from . import units as ureg
from . import DimensionalityError
Quantity = ureg.Quantity
Numbers += (Quantity,)


class Variable(Monomial):
    def __init__(self, *args, **descr):
        """A described singlet Monomial.

        Parameters
        ----------
        *args : list
            may contain "name" (Strings)
                        "value" (Numbers + Quantity) or (Iterable) for a sweep
                        "units" (Strings + Quantity)
                 and/or "label" (Strings)
        **descr : dict
            VarKey description

        Returns
        -------
        Monomials containing a VarKey with the name '$name',
        where $name is the vector's name and i is the VarKey's index.
        """
        for arg in args:
            if isinstance(arg, Strings) and "name" not in descr:
                descr["name"] = arg
            elif isinstance(arg, Numbers) and "value" not in descr:
                descr["value"] = arg
            elif (isinstance(arg, Iterable) and not isinstance(arg, Strings)
                  and "value" not in descr):
                if is_sweepvar(arg):
                    descr["value"] = arg
                else:
                    descr["value"] = ("sweep", arg)
            elif isinstance(arg, Strings+(Quantity,)) and "units" not in descr:
                descr["units"] = arg
            elif isinstance(arg, Strings) and "label" not in descr:
                descr["label"] = arg

        Monomial.__init__(self, **descr)
        self.__class__ = Variable


class VectorVariable(PosyArray):
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
        VarKey description

    Returns
    -------
    PosyArray of Monomials, each containing a VarKey with name '$name_{i}',
    where $name is the vector's name and i is the VarKey's index.
    """

    def __new__(cls, length, *args, **descr):
        cls = PosyArray

        if "idx" in descr:
            raise KeyError("the description field 'idx' is reserved")

        descr["length"] = length

        for arg in args:
            if isinstance(arg, Strings) and "name" not in descr:
                descr["name"] = arg
            elif (isinstance(arg, Iterable) and not isinstance(arg, Strings)
                  and "value" not in descr):
                descr["value"] = arg
            elif isinstance(arg, Strings+(Quantity,)) and "units" not in descr:
                descr["units"] = arg
            elif isinstance(arg, Strings) and "label" not in descr:
                descr["label"] = arg

        values = descr.pop("value", [])
        if len(values) and len(values) != length:
            raise ValueError("values length must be the same as the vector's.")

        if "name" not in descr:
            descr["name"] = "\\fbox{%s}" % VarKey.new_unnamed_id()

        vl = []
        for i in range(length):
            descr.update({"idx": i})
            if len(values):
                descr.update({"value": values[i]})
            vl.append(Variable(**descr))

        obj = np.asarray(vl).view(cls)
        obj.descr = dict(vl[0].exp.keys()[0].descr)
        obj.descr.pop("idx", None)
        obj._hashvalue = hash(VarKey(**obj.descr))

        return obj
