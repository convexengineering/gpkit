import numpy as np
from collections import defaultdict
from itertools import chain
from .small_classes import Numbers, Strings
from .small_scripts import is_sweepvar, veckeyed


class KeyDict(dict):
    """KeyDicts allow storing and accessing the same value with multiple keys

    A KeyDict keeps an internal list of VarKeys as canonical keys, but allows
    accessing their values with any object whose `key` attribute matches
    one of those VarKeys, or with strings who match any of the multiple
    possible string interpretations of each key.

    ```
    kd = gpkit.keydict.KeyDict()
    x = gpkit.Variable("x", model="test")
    kd[x] = 1
    assert kd[x] == kd[x.key] == kd["x"] == kd["x_test"] == 1
    ```

    In addition, if collapse_arrays is True then VarKeys which have a `shape`
    parameter (indicating they are part of an array) are stored as numpy
    arrays, and automatically de-indexed when a matching VarKey with a
    particular `idx` parameter is used as a key.

    ```
    v = gpkit.VectorVariable(3, "v")
    kd[v] = np.array([2, 3, 4])
    assert all(kd[v] == kd[v.key])
    assert all(kd["v"] == np.array([2, 3, 4]))
    assert v[0].key.idx == (0,)
    assert kd[v][0] == kd[v[0]] == 2
    kd[v[0]] = 6
    assert kd[v][0] == kd[v[0]] == 6
    assert all(kd[v] == np.array([6, 3, 4]))
    ```

    By default a KeyDict will regenerate the list of possible key strings
    for every usage; a KeyDict may instead be "baked" to have a fixed list of
    keystrings by calling the `bake()` method.
    """
    collapse_arrays = True

    def __init__(self, *args, **kwargs):
        "Passes through to dict.__init__ via the `update()` method"
        self.baked_keystrs = None
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        "Iterates through the dictionary created by args and kwargs"
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def bake(self):
        "Permanently sets the keystrs of a KeyDict"
        self.baked_keystrs = None
        self.baked_keystrs = self.keystrs()

    def keystrs(self):
        "Generates the strings that may be used as keys for a KeyDict"
        if self.baked_keystrs:
            return self.baked_keystrs
        keystrs = defaultdict(set)
        for key in self.keys():
            for keystr in key.allstrs:
                keystrs[keystr].add(key)
        return keystrs

    @classmethod
    def with_keys(cls, keyset, dictionaries):
        "Generates a KeyDict from a KeySet and iterable of dictionaries"
        out = cls()
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                # The keyset filters and converts each dictionary's keys
                keys = keyset[key]
                for key in keys:
                    if not key.idx:
                        out[key] = value
                    else:
                        if not hasattr(value, "shape"):
                            value = np.array(value)
                        val_i = value[key.idx]
                        if is_sweepvar(val_i) or not np.isnan(val_i):
                            out[key] = val_i
        return out

    def __contains__(self, key):
        "In a winding way, figures out if a key is in the KeyDict"
        if dict.__contains__(self, key):
            return True
        elif hasattr(key, "key"):
            if dict.__contains__(self, key.key):
                return True
            elif self.is_veckey_but_not_collapsed(key):
                if any(k in self for k in self.getkeys(key)):
                    return True
            elif self.is_index_into_vector(key.key):
                vk = veckeyed(key.key)
                if dict.__contains__(self, vk):
                    return not bool(np.isnan(self.__dgi(vk)[key.key.idx]))
        elif key in self.keystrs():
            return True

    def getkeys(self, key):
        "Gets all keys in self that are represented by a given key"
        if isinstance(key, Strings):
            return self.keystrs()[key]
        elif hasattr(key, "key"):
            key = key.key
            if self.is_veckey_but_not_collapsed(key):
                keys = set()
                array = np.empty(key.descr["shape"])
                it = np.nditer(array, flags=['multi_index', 'refs_ok'])
                while not it.finished:
                    i = it.multi_index
                    it.iternext()
                    idx_key = key.__class__(idx=i, **key.descr)
                    keys.add(idx_key)
                return keys
            else:
                return set([key])
        raise ValueError("%s %s is an invalid key for a KeyDict."
                         % (key, type(key)))

    def is_veckey_but_not_collapsed(self, key):
        "True iff the key is a veckey and this KeyDict doesn't collapse arrays"
        if "shape" not in key.descr:
            return False
        if "idx" in key.descr:
            return False
        if self.collapse_arrays:
            return False
        return True

    def is_index_into_vector(self, key):
        "True iff the key indexes into a collapsed array"
        if not self.collapse_arrays:
            return False
        if "idx" not in key.descr:
            return False
        if "shape" not in key.descr:
            return False
        return True

    def __dgi(self, key):
        "Shortcut for calling dict.__getitem__"
        return dict.__getitem__(self, key)

    def __getitem__(self, key):
        "Overloads __getitem__ and [] access to work with all keys"
        keys = self.getkeys(key)
        if len(keys) > 1:
            out = KeyDict()
            out.collapse_arrays = self.collapse_arrays
            for key in keys:
                if self.is_index_into_vector(key):
                    out[key] = self.__dgi(veckeyed(key))[key.descr["idx"]]
                else:
                    out[key] = self.__dgi(key)
        elif keys:
            key, = keys
            if self.is_index_into_vector(key):
                out = self.__dgi(veckeyed(key))[key.descr["idx"]]
            else:
                out = self.__dgi(key)
        else:
            raise KeyError("%s was not found." % key)
        return out

    def __setitem__(self, key, value):
        "Overloads __setitem__ and []= to work with all keys"
        for key in self.getkeys(key):
            if self.is_index_into_vector(key):
                veckey = veckeyed(key)
                if veckey not in self:
                    kwargs = {}
                    if not isinstance(value, Numbers):
                        kwargs["dtype"] = "object"
                    emptyvec = np.full(key.descr["shape"], np.nan, **kwargs)
                    dict.__setitem__(self, veckey, emptyvec)
                self.__dgi(veckey)[key.descr["idx"]] = value
            else:
                dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        "Overloads del [] to work with all keys"
        deletion = False
        for key in self.getkeys(key):
            if dict.__contains__(self, key):
                dict.__delitem__(self, key)
                deletion = True
            elif "shape" in key.descr and "idx" in key.descr:
                self.__dgi(veckeyed(key))[key.descr["idx"]] = np.nan
                if np.isnan(self.__dgi(veckeyed(key))).all():
                    dict.__delitem__(self, veckeyed(key))
                deletion = True
        if not deletion:
            raise KeyError("key %s not found." % key)


class KeySet(KeyDict):
    "KeySets are KeyDicts without values, serving only to filter and map keys"
    collapse_arrays = False

    def __getitem__(self, key):
        "Given a key, returns a list of VarKeys"
        if key not in self:
            return []
        return [k for k in self.getkeys(key) if k in self]

    def __setitem__(self, key, value):
        "Assigns the key itself every time."
        KeyDict.__setitem__(self, key, key)

    def map(self, iterable):
        "Given a list of keys, returns a list of VarKeys"
        varkeys = []
        for key in iterable:
            keys = self[key]
            if len(keys) > 1:
                raise ValueError("KeySet.map() accepts only unambiguous keys.")
            key, = keys
            varkeys.append(key)
        if len(varkeys) == 1:
            varkeys = varkeys[0]
        return varkeys
