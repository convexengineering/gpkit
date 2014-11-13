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


class PosyArray(np.ndarray):
    """A Numpy array with elementwise inequalities and substitutions.

    Parameters
    ----------
    input_array : array-like

    Example
    -------
    >>> px = gpkit.PosyArray([1, x, x**2])
    """

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
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def _latex(self, unused=None):
        "Returns 1D latex list of contents."
        return ("\\begin{bmatrix}" +
                ", & ".join(el._latex() for el in self) +
                "\\end{bmatrix}")

    def __str__(self):
        "Returns list-like string, but with str(el) instead of repr(el)."
        return "["+", ".join([str(p) for p in self])+"]"

    def __repr__(self):
        "Returns str(self) tagged with gpkit information."
        return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))

    def __nonzero__(self):
        "Allows the use of PosyArrays as truth elements."
        return all(p.__nonzero__() for p in self)

    _eq = np.vectorize(lambda a, b: a == b)
    def __eq__(self, other):
        "Applies == in a vectorized fashion."
        return PosyArray([e for e in self._eq(self, other)])

    def __ne__(self, m):
        "Does type checking, then apples 'not ==' in a vectorized fashion."
        return (not isinstance(other, self.__class__)
                and all(self._eq(self, other)))

    # inequality constraints
    _leq = np.vectorize(lambda a, b: a <= b)
    def __le__(self, other):
        "Applies '<=' in a vectorized fashion."
        return PosyArray([e for e in self._leq(self, other)])

    _geq = np.vectorize(lambda a, b: a >= b)
    def __ge__(self, other):
        "Applies '>=' in a vectorized fashion."
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
        return PosyArray([p.sub(subs, val, allow_negative=allow_negative)
                         for p in self])
