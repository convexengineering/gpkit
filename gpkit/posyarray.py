# -*- coding: utf-8 -*-
"""Module for creating PosyArray instances.

    Example
    -------
    >>> x = gpkit.Monomial('x')
    >>> px = gpkit.PosyArray([1, x, x**2])

"""

import numpy as np
from operator import add, mul
from functools import reduce

from . import units as ureg
Quantity = ureg.Quantity


class PosyArray(np.ndarray):
    """A Numpy array with elementwise inequalities and substitutions.

    Parameters
    ----------
    input_array : array-like

    Example
    -------
    >>> px = gpkit.PosyArray([1, x, x**2])
    """

    def __hash__(self):
        if hasattr(self, "_hashvalue"):
            return self._hashvalue
        else:
            return np.ndarray.__hash__(self)

    def __new__(cls, input_array, info=None):
        "Constructor. Required for objects inheriting from np.ndarray."
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        "Finalizer. Required for objects inheriting from np.ndarray."
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

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

    @property
    def c(self):
        try:
            return np.array(self, dtype='float')
        except TypeError:
            raise ValueError("only a posyarray of numbers can be cast to float")

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
        return PosyArray([e for e in self._eq(self, other)])

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
        return PosyArray([e for e in self._leq(self, other)])

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
        return PosyArray([e for e in self._geq(self, other)])

    def outer(self, other):
        "Returns the array and argument's outer product."
        return PosyArray(np.outer(self, other))

    def sum(self):
        "Returns the sum of the array."
        return reduce(add, self[1:], self[0])

    def prod(self):
        "Returns the product of the array."
        return reduce(mul, self[1:], self[0])

    def sub(self, subs, val=None, allow_negative=False):
        "Substitutes into the array"
        return PosyArray([p.sub(subs, val, allow_negative) for p in self])
