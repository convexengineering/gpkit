"""Miscellaneous small classes"""
import numpy as np
from collections import namedtuple as nt
from . import units

try:
    isinstance("", basestring)
    Strings = (str, unicode)
except NameError:
    Strings = (str,)

Quantity = units.Quantity
Numbers = (int, float, np.number, Quantity)

PosyTuple = nt('PosyTuple', ['exps', 'cs', 'varlocs', 'substitutions'])
CootMatrixTuple = nt('CootMatrix', ['row', 'col', 'data'])


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
        self.start = -1

    def __call__(self):
        self.start += 1
        return self.start


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
            enlist_dict(sol, self)
            self.initialized = True
        else:
            append_dict(sol, self)

    def atindex(self, i):
        "Indexes into each list independently."
        return index_dict(i, self, {})

    def toarray(self, shape=None):
        "Converts all lists into arrays."
        if shape is None:
            enray_dict(self, self)


def enlist_dict(i, o):
    "Recursviely copies dict i into o, placing non-dict items into lists."
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = enlist_dict(v, {})
        else:
            o[k] = [v]
    assert set(i.keys()) == set(o.keys())
    return o


def append_dict(i, o):
    "Recursviely travels dict o and appends items found in i."
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = append_dict(v, o[k])
        else:
            o[k].append(v)
    assert set(i.keys()) == set(o.keys())
    return o


def index_dict(idx, i, o):
    "Recursviely travels dict i, placing items at idx into dict o."
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = index_dict(idx, v, {})
        else:
            o[k] = v[idx]
    assert set(i.keys()) == set(o.keys())
    return o


def enray_dict(i, o):
    "Recursively turns lists into numpy arrays."
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = enray_dict(v, {})
        else:
            if len(v) == 1:
                o[k] = np.array(v[0])
            else:
                o[k] = np.array(v)
    assert set(i.keys()) == set(o.keys())
    return o


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
        return HashVector({key: -val for (key, val) in self.items()})

    def __pow__(self, other):
        "Accepts scalars. Return Hashvector with each value put to a power."
        if isinstance(other, Numbers):
            return HashVector({key: val**other for (key, val) in self.items()})
        else:
            return NotImplemented

    def __mul__(self, other):
        """Accepts scalars and dicts. Returns with each value multiplied.

        If the other object inherits from dict, multiplication is element-wise
        and their key's intersection will form the new keys."""
        if isinstance(other, Numbers):
            return HashVector({key: val*other for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self).intersection(other)
            return HashVector({key: self[key] * other[key] for key in keys})
        else:
            return NotImplemented

    def __add__(self, other):
        """Accepts scalars and dicts. Returns with each value added.

        If the other object inherits from dict, addition is element-wise
        and their key's union will form the new keys."""
        if isinstance(other, Numbers):
            return HashVector({key: val+other
                               for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self).union(other)
            sums = {key: self.get(key, 0) + other.get(key, 0) for key in keys}
            return HashVector(sums)
        else:
            return NotImplemented

    def __sub__(self, other): return self + -other
    def __rsub__(self, other): return other + -self
    def __radd__(self, other): return self + other
    def __div__(self, other): return self * other**-1
    def __rdiv__(self, other): return other * self**-1
    def __rmul__(self, other): return self * other
