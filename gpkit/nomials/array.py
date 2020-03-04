"""Module for creating NomialArray instances.

    Example
    -------
    >>> x = gpkit.Monomial('x')
    >>> px = gpkit.NomialArray([1, x, x**2])

"""
from operator import eq, le, ge, xor
from functools import reduce  # pylint: disable=redefined-builtin
import numpy as np
from .map import NomialMap
from ..small_classes import Numbers, HashVector, EMPTY_HV
from ..small_scripts import try_str_without, mag
from ..constraints import ArrayConstraint
from ..repr_conventions import ReprMixin
from ..exceptions import DimensionalityError

@np.vectorize
def vec_recurse(element, function, *args, **kwargs):
    "Vectorizes function with particular args and kwargs"
    return function(element, *args, **kwargs)


def array_constraint(symbol, func):
    "Return function which creates constraints of the given operator."
    vecfunc = np.vectorize(func)

    def wrapped_func(self, other):
        "Creates array constraint from vectorized operator."
        if not isinstance(other, NomialArray):
            other = NomialArray(other)
        result = vecfunc(self, other)
        return ArrayConstraint(result, getattr(self, "key", self),
                               symbol, getattr(other, "key", other))
    return wrapped_func


class NomialArray(ReprMixin, np.ndarray):
    """A Numpy array with elementwise inequalities and substitutions.

    Arguments
    ---------
    input_array : array-like

    Example
    -------
    >>> px = gpkit.NomialArray([1, x, x**2])
    """

    def __mul__(self, other, *, reverse_order=False):
        astorder = (self, other) if not reverse_order else (other, self)
        out = NomialArray(np.ndarray.__mul__(self, other))
        out.ast = ("mul", astorder)
        return out

    def __truediv__(self, other):
        out = NomialArray(np.ndarray.__truediv__(self, other))
        out.ast = ("div", (self, other))
        return out

    def __rtruediv__(self, other):
        out = (np.ndarray.__mul__(self**-1, other))
        out.ast = ("div", (other, self))
        return out

    def __add__(self, other, *, reverse_order=False):
        astorder = (self, other) if not reverse_order else (other, self)
        out = (np.ndarray.__add__(self, other))
        out.ast = ("add", astorder)
        return out

    # pylint: disable=multiple-statements
    def __rmul__(self, other): return self.__mul__(other, reverse_order=True)
    def __radd__(self, other): return self.__add__(other, reverse_order=True)

    def __pow__(self, expo):  # pylint: disable=arguments-differ
        out = (np.ndarray.__pow__(self, expo))  # pylint: disable=too-many-function-args
        out.ast = ("pow", (self, expo))
        return out

    def __neg__(self):
        out = (np.ndarray.__neg__(self))
        out.ast = ("neg", self)
        return out

    def __getitem__(self, idxs):
        out = np.ndarray.__getitem__(self, idxs)
        if not getattr(out, "shape", None):
            return out
        out.ast = ("index", (self, idxs))
        return out

    def str_without(self, excluded=()):
        "Returns string without certain fields (such as 'lineage')."
        if self.ast:
            return self.parse_ast(excluded)
        if hasattr(self, "key"):
            return self.key.str_without(excluded)
        if not self.shape:
            return try_str_without(self.flatten()[0], excluded)

        return "[%s]" % ", ".join(
            [try_str_without(np.ndarray.__getitem__(self, i), excluded)
             for i in range(self.shape[0])])  # pylint: disable=unsubscriptable-object

    def latex(self, excluded=()):
        "Returns latex representation without certain fields."
        units = self.latex_unitstr() if "units" not in excluded else ""
        if hasattr(self, "key"):
            return self.key.latex(excluded) + units
        return np.ndarray.__str__(self)

    def __hash__(self):
        return reduce(xor, map(hash, self.flat), 0)

    def __new__(cls, input_array):
        "Constructor. Required for objects inheriting from np.ndarray."
        # Input is an already formed ndarray instance cast to our class type
        return np.asarray(input_array).view(cls)

    def __array_finalize__(self, obj):
        "Finalizer. Required for objects inheriting from np.ndarray."

    def __array_wrap__(self, out_arr, context=None):  # pylint: disable=arguments-differ
        """Called by numpy ufuncs.
        Special case to avoid creation of 0-dimensional arrays
        See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html"""
        if out_arr.ndim:
            return np.ndarray.__array_wrap__(self, out_arr, context)  # pylint: disable=too-many-function-args
        val = out_arr.item()
        return np.float(val) if isinstance(val, np.generic) else val

    __eq__ = array_constraint("=", eq)
    __le__ = array_constraint("<=", le)
    __ge__ = array_constraint(">=", ge)

    def outer(self, other):
        "Returns the array and argument's outer product."
        return NomialArray(np.outer(self, other))

    def vectorize(self, function, *args, **kwargs):
        "Apply a function to each terminal constraint, returning the array"
        return vec_recurse(self, function, *args, **kwargs)

    def sub(self, subs, require_positive=True):
        "Substitutes into the array"
        return self.vectorize(lambda nom: nom.sub(subs, require_positive))

    @property
    def units(self):
        """units must have same dimensions across the entire nomial array"""
        units = None
        for el in self.flat:  # pylint: disable=not-an-iterable
            el_units = getattr(el, "units", None)
            if units is None:
                units = el_units
            elif ((el_units and units != el_units) or
                  (isinstance(el, Numbers) and not (el == 0 or np.isnan(el)))):
                raise DimensionalityError(el_units, units)
        return units

    def sum(self, *args, **kwargs):  # pylint: disable=arguments-differ
        "Returns a sum. O(N) if no arguments are given."
        if args or kwargs or not self.shape:
            return np.ndarray.sum(self, *args, **kwargs)
        hmap = NomialMap()
        hmap.units = self.units
        it = np.nditer(self, flags=["multi_index", "refs_ok"])
        while not it.finished:
            m = self[it.multi_index]
            it.iternext()
            if isinstance(mag(m), Numbers):
                if mag(m):
                    hmap[EMPTY_HV] = mag(m) + hmap.get(EMPTY_HV, 0)
            else:
                hmap += m.hmap
        out = Signomial(hmap)
        out.ast = ("sum", (self, None))
        return out

    def prod(self, *args, **kwargs):  # pylint: disable=arguments-differ
        "Returns a product. O(N) if no arguments and only contains monomials."
        if args or kwargs:
            return np.ndarray.prod(self, *args, **kwargs)
        c, unitpower = 1.0, 0
        exp = HashVector()
        it = np.nditer(self, flags=["multi_index", "refs_ok"])
        while not it.finished:
            m = self[it.multi_index]
            it.iternext()
            if not hasattr(m, "hmap") and len(m.hmap) == 1:
                return np.ndarray.prod(self, *args, **kwargs)
            c *= mag(m.c)
            unitpower += 1
            exp += m.exp
        hmap = NomialMap({exp: c})
        units = self.units
        hmap.units = units**unitpower if units else None
        out = Signomial(hmap)
        out.ast = ("prod", (self, None))
        return out


from .math import Signomial  # pylint: disable=wrong-import-position
