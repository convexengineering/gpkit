"""Miscellaneous small classes"""
from collections import namedtuple
import numpy as np
from . import units as gpkitunits

try:
    isinstance("", basestring)
    Strings = (str, unicode)
except NameError:
    Strings = (str,)

Quantity = gpkitunits.Quantity
Numbers = (int, float, np.number, Quantity)
CootMatrixTuple = namedtuple('CootMatrix', ['row', 'col', 'data'])


class CootMatrix(CootMatrixTuple):
    "A very simple sparse matrix representation."
    shape = (None, None)

    def append(self, row, col, data):
        if row < 0 or col < 0:
            raise ValueError("Only positive indices allowed")
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def update_shape(self):
        self.shape = (max(self.row)+1, max(self.col)+1)

    def tocoo(self):
        return self.tocsr().tocoo()

    def todense(self):
        return self.tocsr().todense()

    def tocsr(self):
        "Converts to a Scipy sparse csr_matrix"
        from scipy.sparse import csr_matrix
        return csr_matrix((self.data, (self.row, self.col)))

    def tocsc(self):
        return self.tocsr().tocsc()

    def todok(self):
        return self.tocsr().todok()

    def todia(self):
        return self.tocsr().todia()

    def dot(self, arg):
        return self.tocsr().dot(arg)


class Counter(object):
    def __init__(self):
        self.count = -1

    def __call__(self):
        self.count += 1
        return self.count


class SolverLog(list):
    "Adds a `write` method to list so it's file-like and can replace stdout."

    def __init__(self, verbosity=0, output=None, *args, **kwargs):
        super(list, self).__init__(*args, **kwargs)
        self.verbosity = verbosity
        self.output = output

    def write(self, writ):
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
            self.initialized = True
        else:
            _append_dict(sol, self)

    def atindex(self, i):
        "Indexes into each list independently."
        return self.__class__(index_dict(i, self, self.__class__()))

    def to_united_array(self, unitless_keys=(), united=False):
        "Converts all lists into array, potentially grabbing units from keys."
        _enray_and_unit_dict(self, self, unitless_keys, united)


def _enlist_dict(d_in, d_out):
    """Recursviely copies dict d_in into d_out,
    placing non-dict items into lists."""
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _enlist_dict(v, v.__class__())
        else:
            d_out[k] = [v]
    assert set(d_in.keys()) == set(d_out.keys())
    return d_out


def _append_dict(d_in, d_out):
    "Recursviely travels dict d_out and appends items found in d_in."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = _append_dict(v, d_out[k])
        else:
            # consider apennding nan / nanvector for new / missed keys
            d_out[k].append(v)
    # assert set(i.keys()) == set(o.keys())  # keys change with swept varkeys
    return d_out


def index_dict(idx, d_in, d_out):
    "Recursviely travels dict d_in, placing items at idx into dict d_out."
    for k, v in d_in.items():
        if isinstance(v, dict):
            d_out[k] = index_dict(idx, v, v.__class__())
        else:
            try:
                d_out[k] = v[idx]
            except IndexError:  # if not an array, return as is
                d_out[k] = v
    # assert set(i.keys()) == set(o.keys())  # keys change with swept varkeys
    return d_out


def _enray_and_unit_dict(d_in, d_out, unitless_keys=(), united=False):
    "Recursively turns lists into numpy arrays."
    for k, v in d_in.items():
        if isinstance(v, dict):
            if k in unitless_keys:
                united = False
            d_out[k] = _enray_and_unit_dict(v, v.__class__(),
                                            unitless_keys, united)
        else:
            if len(v) == 1:
                v = v[0]
            v = np.array(v)
            if (united and hasattr(k, "units")
                    and isinstance(k.units, Quantity)):
                v = v*k.units
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

    def __init__(self, *args, **kwargs):
        super(HashVector, self).__init__(*args, **kwargs)
        self._hashvalue = None

    def __hash__(self):
        "Allows HashVectors to be used as dictionary keys."
        if self._hashvalue is None:
            self._hashvalue = hash(tuple(self.items()))
        return self._hashvalue

    # temporarily disabling immutability
    #def __setitem__(self, key, value):
    #    raise TypeError("HashVectors are immutable.")

    def __neg__(self):
        "Return Hashvector with each value negated."
        return self.__class__({key: -val for (key, val) in self.items()})

    def __pow__(self, other):
        "Accepts scalars. Return Hashvector with each value put to a power."
        if isinstance(other, Numbers):
            return self.__class__({key: val**other for (key, val) in self.items()})
        else:
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
        else:
            return NotImplemented

    def __add__(self, other):
        """Accepts scalars and dicts. Returns with each value added.

        If the other object inherits from dict, addition is element-wise
        and their key's union will form the new keys."""
        if isinstance(other, Numbers):
            return self.__class__({key: val+other
                                   for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self).union(other)
            sums = {key: self.get(key, 0) + other.get(key, 0) for key in keys}
            return self.__class__(sums)
        else:
            return NotImplemented

    # pylint: disable=multiple-statements
    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return other + -self
    def __radd__(self, other): return self + other
    def __div__(self, other): return self * other**-1
    def __rdiv__(self, other): return other * self**-1
    def __rmul__(self, other): return self * other
