"Implement Variable and ArrayVariable classes"
from collections.abc import Iterable
import numpy as np
from .data import NomialData
from .array import NomialArray
from .math import Monomial
from ..globals import NamedVariables, Vectorize
from ..varkey import VarKey
from ..small_classes import Strings, Numbers
from ..small_scripts import is_sweepvar


def addmodelstodescr(descr, addtonamedvars=None):
    "Add models to descr, optionally adding the second argument to NAMEDVARS"
    if NamedVariables.lineage:
        descr["lineage"] = NamedVariables.lineage
        if addtonamedvars:
            NamedVariables.namedvars[descr["lineage"]].append(addtonamedvars)


class Variable(Monomial):
    """A described singlet Monomial.

    Arguments
    ---------
    *args : list
        may contain "name" (Strings)
                    "value" (Numbers + Quantity) or (Iterable) for a sweep
                    "units" (Strings)
             and/or "label" (Strings)
    **descr : dict
        VarKey description

    Returns
    -------
    Monomials containing a VarKey with the name '$name',
    where $name is the vector's name and i is the VarKey's index.
    """
    def __init__(self, *args, **descr):
        if len(args) == 1 and isinstance(args[0], VarKey):
            self.key, = args
        else:
            for arg in args:
                if isinstance(arg, Strings) and "name" not in descr:
                    descr["name"] = arg
                elif (isinstance(arg, Numbers) or hasattr(arg, "__call__")
                      and "value" not in descr):
                    descr["value"] = arg
                elif (isinstance(arg, Iterable)
                      and not isinstance(arg, Strings)):
                    if is_sweepvar(arg):
                        descr["value"] = arg
                    else:
                        descr["value"] = ("sweep", arg)
                elif isinstance(arg, Strings) and "units" not in descr:
                    descr["units"] = arg
                elif isinstance(arg, Strings) and "label" not in descr:
                    descr["label"] = arg
            addmodelstodescr(descr, addtonamedvars=self)
            self.key = VarKey(**descr)
        Monomial.__init__(self, self.key.hmap)
        # NOTE: needed because Signomial.__init__ will change the class
        self.__class__ = Variable

    __hash__ = NomialData.__hash__

    def to(self, units):
        "Create new Signomial converted to new units"
        return Monomial(self).to(units)  # pylint: disable=no-member

    def sub(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Same as nomial substitution, but also allows single-argument calls

        Example
        -------
        x = Variable('x')
        assert x.sub(3) == Variable('x', value=3)
        """
        if len(args) == 1 and "val" not in kwargs:
            arg, = args
            if not isinstance(arg, dict):
                args = [{self: arg}]
        return Monomial.sub(self, *args, **kwargs)


class ArrayVariable(NomialArray):  # pylint: disable=too-many-locals
    """A described vector of singlet Monomials.

    Arguments
    ---------
    shape : int or tuple
        length or shape of resulting array
    *args :
        may contain "name" (Strings)
                    "value" (Iterable)
                    "units" (Strings)
             and/or "label" (Strings)
    **descr :
        VarKey description

    Returns
    -------
    NomialArray of Monomials, each containing a VarKey with name '$name_{i}',
    where $name is the vector's name and i is the VarKey's index.
    """
    def __new__(cls, shape, *args, **descr):  # pylint: disable=too-many-branches, too-many-statements, arguments-differ
        if "idx" in descr:
            raise ValueError("the description field 'idx' is reserved")

        shape = (shape,) if isinstance(shape, Numbers) else tuple(shape)
        if Vectorize.vectorization:
            shape += Vectorize.vectorization

        descr["shape"] = shape

        for arg in args:
            if isinstance(arg, Strings) and "name" not in descr:
                descr["name"] = arg
            elif (isinstance(arg, (Numbers, Iterable))
                  and not isinstance(arg, Strings)
                  and "value" not in descr):
                descr["value"] = arg
            elif hasattr(arg, "__call__"):
                descr["value"] = arg
            elif isinstance(arg, Strings) and "units" not in descr:
                descr["units"] = arg
            elif isinstance(arg, Strings) and "label" not in descr:
                descr["label"] = arg

        if "name" not in descr:
            descr["name"] = "\\fbox{%s}" % VarKey.unique_id()

        value_option = None
        if "value" in descr:
            value_option = "value"
        if value_option:
            values = descr.pop(value_option)
        if value_option and not hasattr(values, "__call__"):
            if Vectorize.vectorization:
                if not hasattr(values, "shape"):
                    values = np.full(shape, values, "f")
                else:
                    values = np.broadcast_to(values, reversed(shape)).T
            elif not hasattr(values, "shape"):
                values = np.array(values)
            if values.shape != shape:
                raise ValueError("the value's shape %s is different than"
                                 " the vector's %s." % (values.shape, shape))

        veckeydescr = descr.copy()
        addmodelstodescr(veckeydescr)
        if value_option:
            if hasattr(values, "__call__"):
                veckeydescr["original_fn"] = values
            veckeydescr[value_option] = values
        veckey = VarKey(**veckeydescr)

        descr["veckey"] = veckey
        vl = np.empty(shape, dtype="object")
        it = np.nditer(vl, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            i = it.multi_index
            it.iternext()
            descr["idx"] = i
            if value_option:
                if hasattr(values, "__call__"):
                    descr[value_option] = veclinkedfn(values, i)
                else:
                    descr[value_option] = values[i]
            vl[i] = Variable(**descr)

        obj = np.asarray(vl).view(NomialArray)
        obj.key = veckey
        obj.units = obj.key.units
        return obj


def veclinkedfn(linkedfn, i):
    "Generate an indexed linking function."
    def newlinkedfn(c):
        "Linked function that pulls out a particular index"
        return np.array(linkedfn(c))[i]
    return newlinkedfn


class VectorizableVariable(Variable, ArrayVariable):  # pylint: disable=too-many-ancestors
    "A Variable outside a vectorized environment, an ArrayVariable within."
    def __new__(cls, *args, **descr):
        if Vectorize.vectorization:
            shape = descr.pop("shape", ())
            return ArrayVariable.__new__(cls, shape, *args, **descr)
        return Variable(*args, **descr)
