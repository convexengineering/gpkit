"""Miscellaneous small classes"""
from operator import xor
import numpy as np
from ._pint import Quantity, qty  # pylint: disable=unused-import
from functools import reduce  # pylint: disable=redefined-builtin

try:
    isinstance("", basestring)
    Strings = (str, unicode)
except NameError:
    Strings = (str,)

Numbers = (int, float, np.number, Quantity)


class Count(object):
    "Like python 2's itertools.count, for Python 3 compatibility."

    def __init__(self):
        self.count = -1

    def next(self):
        "Increment self.count and return it"
        self.count += 1
        return self.count


def matrix_converter(name):
    "Generates conversion function."
    def to_(self):  # used in tocoo, tocsc, etc below
        "Converts to another type of matrix."
        # pylint: disable=unused-variable
        return getattr(self.tocsr(), "to"+name)()
    return to_


class CootMatrix(object):
    "A very simple sparse matrix representation."
    def __init__(self, row, col, data):
        self.row, self.col, self.data = row, col, data
        self.shape = [(max(self.row) + 1) if self.row else 0,
                      (max(self.col) + 1) if self.col else 0]

    def __eq__(self, other):
        return (self.row == other.row and self.col == other.col
                and self.data == other.data and self.shape == other.shape)

    def append(self, row, col, data):
        "Appends entry to matrix."
        if row < 0 or col < 0:
            raise ValueError("Only positive indices allowed")
        if row >= self.shape[0]:
            self.shape[0] = row + 1
        if col >= self.shape[1]:
            self.shape[1] = col + 1
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    tocoo = matrix_converter("coo")
    tocsc = matrix_converter("csc")
    todia = matrix_converter("dia")
    todok = matrix_converter("dok")
    todense = matrix_converter("dense")

    def tocsr(self):
        "Converts to a Scipy sparse csr_matrix"
        from scipy.sparse import csr_matrix
        return csr_matrix((self.data, (self.row, self.col)))

    def dot(self, arg):
        "Returns dot product with arg."
        return self.tocsr().dot(arg)


class SolverLog(list):
    "Adds a `write` method to list so it's file-like and can replace stdout."
    def __init__(self, verbosity=0, output=None, **kwargs):
        list.__init__(self, **kwargs)
        self.verbosity = verbosity
        self.output = output

    def write(self, writ):
        "Append and potentially write the new line."
        if writ != "\n":
            writ = writ.rstrip("\n")
            self.append(writ)
        if self.verbosity > 0:
            self.output.write(writ)


class DictOfLists(dict):
    "A hierarchy of dicionaries, with lists at the bottom."

    def append(self, sol):
        "Appends a dict (of dicts) of lists to all held lists."
        if not hasattr(self, 'initialized'):
            _enlist_dict(sol, self)
            # pylint: disable=attribute-defined-outside-init
            self.initialized = True
        else:
            _append_dict(sol, self)

    def atindex(self, i):
        "Indexes into each list independently."
        return self.__class__(_index_dict(i, self, self.__class__()))

    def to_arrays(self):
        "Converts all lists into array."
        _enray(self, self)


def _enlist_dict(d_in, d_out):
    "Recursively copies d_in into d_out, placing non-dict items into lists."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _enlist_dict(v, v.__class__())
        else:
            d_out[k] = [v]
    assert set(d_in.keys()) == set(d_out.keys())
    return d_out


def _append_dict(d_in, d_out):
    "Recursively travels dict d_out and appends items found in d_in."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _append_dict(v, d_out[k])
        else:
            # consider appending nan / nanvector for new / missed keys
            d_out[k].append(v)
    # assert set(i.keys()) == set(o.keys())  # keys change with swept varkeys
    return d_out


def _index_dict(idx, d_in, d_out):
    "Recursively travels dict d_in, placing items at idx into dict d_out."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _index_dict(idx, v, v.__class__())
        else:
            try:
                d_out[k] = v[idx]
            except IndexError:  # if not an array, return as is
                d_out[k] = v
    # assert set(i.keys()) == set(o.keys())  # keys change with swept varkeys
    return d_out


def _enray(d_in, d_out):
    "Recursively turns lists into numpy arrays."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _enray(v, v.__class__())
        else:
            if len(v) == 1:
                v = v[0]
            else:
                v = np.array(v)
            d_out[k] = v
    # assert set(i.keys()) == set(o.keys())  # keys change with swept varkeys
    return d_out


class HashVector(dict):
    """A simple, sparse, string-indexed vector. Inherits from dict.

    The HashVector class supports element-wise arithmetic:
    any undeclared variables are assumed to have a value of zero.

    Arguments
    ---------
    arg : iterable

    Example
    -------
    >>> x = gpkit.nomials.Monomial('x')
    >>> exp = gpkit.small_classes.HashVector({x: 2})
    """
    def copy(self):
        "Return a copy of this"
        return self.__class__(super(HashVector, self).copy())

    def __hash__(self):
        "Allows HashVectors to be used as dictionary keys."
        # pylint:disable=access-member-before-definition, attribute-defined-outside-init
        if not hasattr(self, "_hashvalue") or self._hashvalue is None:
            self._hashvalue = reduce(xor, map(hash, self.items()), 0)
        return self._hashvalue

    def __neg__(self):
        "Return Hashvector with each value negated."
        return self.__class__({key: -val for (key, val) in self.items()})

    def __pow__(self, other):
        "Accepts scalars. Return Hashvector with each value put to a power."
        if isinstance(other, Numbers):
            return self.__class__({key: val**other
                                   for (key, val) in self.items()})
        return NotImplemented

    def __mul__(self, other):
        """Accepts scalars and dicts. Returns with each value multiplied.

        If the other object inherits from dict, multiplication is element-wise
        and their key's intersection will form the new keys."""
        if isinstance(other, Numbers):
            return self.__class__({key: val*other
                                   for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self).intersection(other)
            return self.__class__({key: self[key] * other[key] for key in keys})
        return NotImplemented

    def __add__(self, other):
        """Accepts scalars and dicts. Returns with each value added.

        If the other object inherits from dict, addition is element-wise
        and their key's union will form the new keys."""
        if isinstance(other, Numbers):
            return self.__class__({key: val+other
                                   for (key, val) in self.items()})
        elif isinstance(other, dict):
            sums = self.copy()
            for key, value in other.items():
                if key in sums:
                    svalue = sums[key]
                    if value == -svalue:
                        del sums[key]  # remove zeros created by addition
                    else:
                        sums[key] = value + svalue
                else:
                    sums[key] = value
            return sums
        return NotImplemented

    # pylint: disable=multiple-statements
    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return other + -self
    def __radd__(self, other): return self + other
    def __div__(self, other): return self * other**-1
    def __rdiv__(self, other): return other * self**-1
    def __rmul__(self, other): return self * other
