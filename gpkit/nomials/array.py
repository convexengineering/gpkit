# -*coding: utf-8 -*-
"""Module for creating NomialArray instances.

    Example
    -------
    >>> x = gpkit.Monomial('x')
    >>> px = gpkit.NomialArray([1, x, x**2])

"""
from operator import eq, le, ge, xor
import numpy as np
from ..small_classes import Numbers
from ..small_scripts import try_str_without
from ..constraints.array import ArrayConstraint
from ..repr_conventions import _str, _repr, _repr_latex_
from .. import units as ureg
from .. import DimensionalityError
Quantity = ureg.Quantity


@np.vectorize
def vec_recurse(element, function, *args, **kwargs):
    "Vectorizes function with particular args and kwargs"
    return function(element, *args, **kwargs)


def array_constraint(symbol, func):
    "Return function which creates constraints of the given operator."
    vecfunc = np.vectorize(func)

    def wrapped_func(self, other):
        "Creates array constraint from vectorized operator."
        if isinstance(other, Quantity):
            veccable_other = np.array([other], dtype=object)
        else:
            veccable_other = other
        result = vecfunc(self, veccable_other)
        left = self.key if hasattr(self, "key") else self
        right = other.key if hasattr(other, "key") else other
        return ArrayConstraint(result, left, symbol, right)
    return wrapped_func


class NomialArray(np.ndarray):
    """A Numpy array with elementwise inequalities and substitutions.

    Arguments
    ---------
    input_array : array-like

    Example
    -------
    >>> px = gpkit.NomialArray([1, x, x**2])
    """

    __str__ = _str
    __repr__ = _repr
    _repr_latex_ = _repr_latex_

    def str_without(self, excluded=()):
        "Returns string without certain fields (such as 'models')."
        if self.shape:
            return "[" + ", ".join([try_str_without(el, excluded)
                                    for el in self]) + "]"
        else:
            return str(self.flatten()[0])  # TODO THIS IS WEIRD

    def latex(self, matwrap=True):
        "Returns 1D latex list of contents."
        if len(self.shape) == 0:
            return self.flatten()[0].latex()
        if len(self.shape) == 1:
            return (("\\begin{bmatrix}" if matwrap else "") +
                    " & ".join(el.latex() for el in self) +
                    ("\\end{bmatrix}" if matwrap else ""))
        elif len(self.shape) == 2:
            return ("\\begin{bmatrix}" +
                    " \\\\\n".join(el.latex(matwrap=False) for el in self) +
                    "\\end{bmatrix}")
        else:
            return None

    def __hash__(self):
        return hash(reduce(xor, map(hash, self.flat), 0))

    def __new__(cls, input_array):
        "Constructor. Required for objects inheriting from np.ndarray."
        # Input array is an already formed ndarray instance
        # cast to be our class type
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        "Finalizer. Required for objects inheriting from np.ndarray."
        pass

    def __array_wrap__(self, out_arr, context=None):
        """Called by numpy ufuncs.
        Special case to avoid creation of 0-dimensional arrays
        See http://docs.scipy.org/doc/numpy/user/basics.subclassing.html"""
        if out_arr.ndim:
            return np.ndarray.__array_wrap__(self, out_arr, context)
        try:
            val = out_arr.item()
            return np.float(val) if isinstance(val, np.generic) else val
        except:
            print("Something went wrong. I'd like to raise a RuntimeWarning,"
                  " but you wouldn't see it because numpy seems to catch all"
                  " Exceptions coming from __array_wrap__.")
            raise

    def __nonzero__(self):
        "Allows the use of NomialArrays as truth elements."
        return all(bool(p) for p in self.flat)

    def __bool__(self):
        "Allows the use of NomialArrays as truth elements in python3."
        return all(bool(p) for p in self.flat)

    @property
    def c(self):
        """The coefficient vector in the GP input data sense"""
        try:
            floatarray = np.array(self, dtype='float')
            if not floatarray.shape:
                return floatarray.flatten()[0]
            else:
                return floatarray
        except TypeError:
            raise ValueError("only a nomialarray of numbers has a 'c'")

    def vectorize(self, function, *args, **kwargs):
        "Apply a function to each terminal constraint, returning the array"
        return vec_recurse(self, function, *args, **kwargs)

    __eq__ = array_constraint("=", eq)
    __le__ = array_constraint("<=", le)
    __ge__ = array_constraint(">=", ge)

    def __ne__(self, other):
        "Does type checking, then applies 'not ==' in a vectorized fashion."
        return not isinstance(other, self.__class__) or not all(self == other)

    def outer(self, other):
        "Returns the array and argument's outer product."
        return NomialArray(np.outer(self, other))

    def sub(self, subs, val=None, require_positive=True):
        "Substitutes into the array"
        return self.vectorize(lambda nom: nom.sub(subs, val, require_positive))

    @property
    def units(self):
        """units must have same dimensions across the entire nomial array"""
        units = None
        for el in self.flat:
            if isinstance(el, Numbers):
                if units and not (el == 0 or np.isnan(el)):
                    raise DimensionalityError("elements of a NomialArray"
                                              "must have the same units.")
            elif units:
                try:
                    (units/el.units).to("dimensionless")
                except DimensionalityError:
                    raise DimensionalityError("elements of a NomialArray"
                                              "must have the same units.")
                except:
                    raise RuntimeWarning("%s %s" % (type(el), el))
            else:
                units = el.units
        return units

    def padleft(self, padding):
        "Returns ({padding}, self[0], self[1] ... self[N])"
        if self.ndim != 1:
            raise NotImplementedError("unimplemented for ndim=%s" % self.ndim)
        padded = NomialArray(np.hstack((padding, self)))
        _ = padded.units  # check that the units are consistent
        return padded

    def padright(self, padding):
        "Returns (self[0], self[1] ... self[N], {padding})"
        if self.ndim != 1:
            raise NotImplementedError("unimplemented for ndim=%s" % self.ndim)
        padded = NomialArray(np.hstack((self, padding)))
        _ = padded.units  # check that the units are consistent
        return padded

    @property
    def left(self):
        "Returns (0, self[0], self[1] ... self[N-1])"
        return self.padleft(0)[:-1]

    @property
    def right(self):
        "Returns (self[1], self[2] ... self[N], 0)"
        return self.padright(0)[1:]
