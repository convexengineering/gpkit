import numpy as np

from collections import namedtuple as nt

try:
    isinstance("", basestring)
    Strings = (str, unicode)
except NameError:
    Strings = (str,)


Numbers = (int, float, np.number)

PosyTuple = nt('PosyTuple', ['exps', 'cs', 'varlocs', 'substitutions'])
CootMatrixTuple = nt('CootMatrix', ['row', 'col', 'data'])


class count(object):

    def __init__(self):
        self.start = -1

    def __call__(self):
        self.start += 1
        return self.start


class CootMatrix(CootMatrixTuple):
    "A very simple sparse matrix representation."
    shape = (None, None)

    def append(self, i, j, x):
        assert (i >= 0 and j >= 0), "Only positive indices allowed"
        self.row.append(i)
        self.col.append(j)
        self.data.append(x)

    def update_shape(self):
        self.shape = (max(self.row)+1, max(self.col)+1)

    def tocoo(self):
        "Converts to a Scipy sparse coo_matrix"
        from scipy.sparse import coo_matrix
        return coo_matrix((self.data, (self.row, self.col)))

    def todense(self): return self.tocoo().todense()
    def tocsr(self):   return self.tocoo().tocsr()
    def tocsc(self):   return self.tocoo().tocsc()
    def todok(self):   return self.tocoo().todok()
    def todia(self):   return self.tocoo().todia()


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

    Parameters
    ----------
    arg : iterable

    Example
    -------
    >>> x = gpkit.nomials.Monomial('x')
    >>> exp = gpkit.small_classes.HashVector({x: 2})
    """

    def __hash__(self):
        "Allows HashVectors to be used as dictionary keys."
        if not hasattr(self, "_hashvalue"):
            self._hashvalue = hash(tuple(self.items()))
        return self._hashvalue

    # temporarily disabling immutability
    #def __setitem__(self, key, value):
    #    raise TypeError("HashVectors are immutable.")

    def __neg__(self):
        "Return Hashvector with each value negated."
        return HashVector({key: -val for (key, val) in self.items()})

    def __pow__(self, x):
        "Accepts scalars. Return Hashvector with each value put to a power."
        if isinstance(other, Numbers):
            return HashVector({key: val**x for (key, val) in self.items()})
        else:
            return NotImplemented

    def __mul__(self, other):
        """Accepts scalars and dicts. Returns with each value multiplied.

        If the other object inherits from dict, multiplication is element-wise
        and their key's intersection will form the new keys."""
        if isinstance(other, Numbers):
            return HashVector({key: val*other for (key, val) in self.items()})
        elif isinstance(other, dict):
            keys = set(self.keys()).intersection(other.keys())
            sums = {key: self[key] * other[key] for key in keys}
            return HashVector(sums)
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
            keys = set(self.keys()).union(other.keys())
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
