"Implement Variable and ArrayVariable classes"
from collections import Iterable
import numpy as np
from .data import NomialData
from .array import NomialArray
from .nomial_math import Monomial
from ..varkey import VarKey
from ..small_classes import Strings, Numbers, Quantity
from ..small_scripts import is_sweepvar


class Variable(Monomial):
    """A described singlet Monomial.

    Arguments
    ---------
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

    def __init__(self, *args, **descr):
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

        if descr.pop("newvariable", True):
            from .. import MODELS, MODELNUMS, NAMEDVARS

            if MODELS and MODELNUMS:
                NAMEDVARS[tuple(MODELS), tuple(MODELNUMS)].append(self)

            if MODELS:
                descr["models"] = descr.get("models", []) + MODELS
            if MODELNUMS:
                descr["modelnums"] = descr.get("modelnums", []) + MODELNUMS

        Monomial.__init__(self, **descr)
        # NOTE: because Signomial.__init__ will change the class
        self.__class__ = Variable

    __hash__ = NomialData.__hash__

    @property
    def key(self):
        """Get the VarKey associated with this Variable"""
        return list(self.exp)[0]

    @property
    def descr(self):
        "a Variable's descr is derived from its VarKey."
        return self.key.descr

    def sub(self, *args, **kwargs):
        """Same as nomial substitution, but also allows single-argument calls

        Example
        -------
        x = Variable('x')
        assert x.sub(3) == Variable('x', value=3)
        """
        if len(args) == 1 and "val" not in kwargs:
            arg, = args
            if not isinstance(arg, dict):
                args = ({self: arg},)
        return Monomial.sub(self, *args, **kwargs)


class ArrayVariable(NomialArray):
    """A described vector of singlet Monomials.

    Arguments
    ---------
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
    NomialArray of Monomials, each containing a VarKey with name '$name_{i}',
    where $name is the vector's name and i is the VarKey's index.
    """

    def __new__(cls, shape, *args, **descr):
        # pylint: disable=too-many-branches, too-many-statements
        cls = NomialArray

        if "idx" in descr:
            raise KeyError("the description field 'idx' is reserved")

        shape = (shape,) if isinstance(shape, Numbers) else tuple(shape)
        from .. import VECTORIZATION
        if VECTORIZATION:
            shape = shape + tuple(VECTORIZATION)
            descr["vectorization"] = VECTORIZATION

        descr["shape"] = shape

        for arg in args:
            if isinstance(arg, Strings) and "name" not in descr:
                descr["name"] = arg
            elif (isinstance(arg, (Numbers, Iterable))
                  and not isinstance(arg, Strings)
                  and "value" not in descr):
                descr["value"] = arg
            elif isinstance(arg, Strings+(Quantity,)) and "units" not in descr:
                descr["units"] = arg
            elif isinstance(arg, Strings) and "label" not in descr:
                descr["label"] = arg

        values, value_option = [], None
        for value_option in ["value", "sp_init"]:
            if value_option in descr:
                values = descr.pop(value_option)
                if "vectorization" in descr:
                    if shape:
                        values = np.full(shape, values, "f")
                    else:
                        descr["value"] = values
                        values = []
                break

        descr.pop("vectorization", None)

        valuetype = ""
        if len(values):
            if len(shape) == 1:
                shape_match = len(values) == shape[0]
                valuetype = "list"
            else:
                values = np.array(values)    # pylint: disable=redefined-variable-type
                shape_match = values.shape == shape
                valuetype = "array"
            if not shape_match:
                raise ValueError("the value's shape must be the same"
                                 " as the vector's.")

        if "name" not in descr:
            descr["name"] = "\\fbox{%s}" % VarKey.new_unnamed_id()

        arraykey = VarKey(**descr)
        vl = np.empty(shape, dtype="object")
        it = np.nditer(vl, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            i = it.multi_index
            it.iternext()
            descr.update({"idx": i})
            if valuetype == "array":
                descr.update({value_option: values[i]})
            elif valuetype == "list":
                descr.update({value_option: values[i[0]]})
            vl[i] = Variable(**descr)
            vl[i].key.arraykey = arraykey

        obj = np.asarray(vl).view(cls)
        obj.descr = descr
        obj.descr.pop("idx", None)
        obj.key = VarKey(**obj.descr)

        return obj


# pylint: disable=too-many-ancestors
class VectorizableVariable(Variable, ArrayVariable):
    "A Variable outside a vectorized environment, an ArrayVariable within."
    def __new__(cls, *args, **descr):
        from .. import VECTORIZATION
        if VECTORIZATION:
            shape = descr.pop("shape", ())
            return ArrayVariable.__new__(cls, shape, *args, **descr)
        else:
            return Variable(*args, **descr)
