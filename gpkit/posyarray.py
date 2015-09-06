# -*coding: utf-8 -*-
"""Module for creating PosyArray instances.

    Example
    -------
    >>> x = gpkit.Monomial('x')
    >>> px = gpkit.PosyArray([1, x, x**2])

"""

import numpy as np

from .nomials import Monomial
from .small_classes import Numbers

from . import units as ureg
from . import DimensionalityError
Quantity = ureg.Quantity


class PosyArray(np.ndarray):
    """A Numpy array with elementwise inequalities and substitutions.

    Arguments
    ---------
    input_array : array-like

    Example
    -------
    >>> px = gpkit.PosyArray([1, x, x**2])
    """

    def __str__(self):
        "Returns list-like string, but with str(el) instead of repr(el)."
        return "[" + ", ".join(str(p) for p in self) + "]"

    def __repr__(self):
        "Returns str(self) tagged with gpkit information."
        return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))

    def __hash__(self):
        return getattr(self, "_hashvalue", np.ndarray.__hash__(self))

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

    def _latex(self, unused=None, matwrap=True):
        "Returns 1D latex list of contents."
        if len(self.shape) == 1:
            return (("\\begin{bmatrix}" if matwrap else "") +
                    " & ".join(el._latex() for el in self) +
                    ("\\end{bmatrix}" if matwrap else ""))
        elif len(self.shape) == 2:
            return ("\\begin{bmatrix}" +
                    " \\\\\n".join(el._latex(matwrap=False) for el in self) +
                    "\\end{bmatrix}")
        else:
            return None

    def __nonzero__(self):
        "Allows the use of PosyArrays as truth elements."
        return all(p.__nonzero__() for p in self)

    def __bool__(self):
        "Allows the use of PosyArrays as truth elements in python3."
        return all(p.__bool__() for p in self)

    @property
    def c(self):
        try:
            return np.array(self, dtype='float')
        except TypeError:
            raise ValueError("only a posyarray of numbers has a 'c'")

    _eq = np.vectorize(lambda a, b: a == b)

    def __eq__(self, other):
        "Applies == in a vectorized fashion."
        if isinstance(other, Quantity):
            if isinstance(other.magnitude, np.ndarray):
                l = []
                for i, e in enumerate(self):
                    l.append(e == other[i])
                return PosyArray(l)
            else:
                return PosyArray([e == other for e in self])
        return PosyArray(self._eq(self, other))

    def __ne__(self, m):
        "Does type checking, then applies 'not ==' in a vectorized fashion."
        return (not isinstance(other, self.__class__)
                or not all(self._eq(self, other)))

    # inequality constraints
    _leq = np.vectorize(lambda a, b: a <= b)

    def __le__(self, other):
        "Applies '<=' in a vectorized fashion."
        if isinstance(other, Quantity):
            if isinstance(other.magnitude, np.ndarray):
                l = []
                for i, e in enumerate(self):
                    l.append(e <= other[i])
                return PosyArray(l)
            else:
                return PosyArray([e <= other for e in self])
        return PosyArray(self._leq(self, other))

    _geq = np.vectorize(lambda a, b: a >= b)

    def __ge__(self, other):
        "Applies '>=' in a vectorized fashion."
        if isinstance(other, Quantity):
            if isinstance(other.magnitude, np.ndarray):
                l = []
                for i, e in enumerate(self):
                    l.append(e >= other[i])
                return PosyArray(l)
            else:
                return PosyArray([e >= other for e in self])
        return PosyArray(self._geq(self, other))

    def outer(self, other):
        "Returns the array and argument's outer product."
        return PosyArray(np.outer(self, other))

    def sub(self, subs, val=None, require_positive=True):
        "Substitutes into the array"
        return PosyArray([p.sub(subs, val, require_positive) for p in self])

    @property
    def units(self):
        units = None
        for el in self:
            if not isinstance(el, Numbers) or el != 0 and not np.isnan(el):
                if units:
                    try:
                        (units/el.units).to("dimensionless")
                    except DimensionalityError:
                        raise ValueError("all elements of a PosyArray must"
                                         " have the same units.")
                else:
                    units = el.units
        return units

    @property
    def padleft(self):
        "Returns (0, self[0], self[1] ... self[N])"
        return PosyArray(np.hstack((fencepost_zero(self), self)))

    @property
    def padright(self):
        "Returns (self[0], self[1] ... self[N], 0)"
        return PosyArray(np.hstack((self, fencepost_zero(self))))

    @property
    def left(self):
        "Returns (0, self[0], self[1] ... self[N-1])"
        return self.padleft[:-1]

    @property
    def right(self):
        "Returns (self[1], self[2] ... self[N], 0)"
        return self.padright[1:]


def fencepost_zero(array):
    if array.ndim != 1:
        raise NotImplementedError("not implemented for ndim = %s" %
                                  array.ndim)
    zero = Monomial(0, units=array.units, require_positive=False)
    zero.fencepost = ["pgt"]  # no reason to plt fence a zero
    return zero
