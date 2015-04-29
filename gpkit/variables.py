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
            elif (((isinstance(arg, Iterable) and not isinstance(arg, Strings))
                  or hasattr(arg, "__call__")) and "value" not in descr):
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

    @property
    def descr(self):
        return list(self.exp)[0].descr


class VectorVariable(PosyArray):
    """A described vector of singlet Monomials.

    Parameters
    ----------
    shape : int or tuple
        length or shape of resulting array
    *args :
        may contain "name" (Strings)
                    "value" (Iterable)
                    "units" (Strings + Quantity)
             and/or "label" (Strings)
    **descr :
        VarKey description

    Returns
    -------
    PosyArray of Monomials, each containing a VarKey with name '$name_{i}',
    where $name is the vector's name and i is the VarKey's index.
    """

    def __new__(cls, shape, *args, **descr):
        cls = PosyArray

        if "idx" in descr:
            raise KeyError("the description field 'idx' is reserved")

        if isinstance(shape, Numbers):
            shape = (shape,)

        descr["shape"] = shape

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
        valuetype = ""
        if len(values):
            if hasattr(values, "shape"):
                shape_match = values.shape == shape
                valuetype = "array"
            else:
                shape_match = len(shape) == 1 and len(values) == shape[0]
                valuetype = "list"
            if not shape_match:
                raise ValueError("the value's shape must be the same"
                                 " as the vector's.")

        if "name" not in descr:
            descr["name"] = "\\fbox{%s}" % VarKey.new_unnamed_id()

        vl = []
        it = np.nditer(np.empty(shape), flags=['multi_index', 'refs_ok'])
        while not it.finished:
            i = it.multi_index
            it.iternext()
            descr.update({"idx": i})
            if valuetype == "array":
                descr.update({"value": values[i]})
            elif valuetype == "list":
                descr.update({"value": values[i[0]]})
            vl.append(Variable(**descr))

        obj = np.asarray(vl).view(cls)
        obj.descr = dict(list(vl[0].exp)[0].descr)
        obj.descr.pop("idx", None)
        obj._hashvalue = hash(VarKey(**obj.descr))

        return obj


ArrayVariable = VectorVariable
