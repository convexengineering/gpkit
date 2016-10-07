"Implements KeyDict and KeySet classes"
from collections import defaultdict
import numpy as np
from .small_classes import Numbers


class KeyDict(dict):
    """KeyDicts allow storing and accessing the same value with multiple keys

    A KeyDict keeps an internal list of VarKeys as canonical keys, but allows
    accessing their values with any object whose `key` attribute matches
    one of those VarKeys, or with strings who match any of the multiple
    possible string interpretations of each key.

    Creating a KeyDict:
    >>>> kd = gpkit.keydict.KeyDict()

    Now kd[x] can be set, where x is any gpkit Variable or VarKey.
    __getitem__ is such that kd[x] can be accessed using:
     - x
     - x.key
     - x.name (a string)
     - "x_modelname" (x's name including modelname)

    In addition, if collapse_arrays is True then VarKeys which have a `shape`
    parameter (indicating they are part of an array) are stored as numpy
    arrays, and automatically de-indexed when a matching VarKey with a
    particular `idx` parameter is used as a key.

    Note that if a item is set using a key that does not have a `.key`
    attribute, that key can be set and accessed normally.

    See also: gpkit/tests/t_keydict.py.
    """
    collapse_arrays = True

    def __init__(self, *args, **kwargs):
        "Passes through to dict.__init__ via the `update()` method"
        # pylint: disable=super-init-not-called
        self.varkeys = None
        self.keymap = defaultdict(set)
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        "Iterates through the dictionary created by args and kwargs"
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def parse_and_index(self, key):
        "Returns key if key had one, and veckey/idx for indexed veckeys."
        if hasattr(key, "key"):
            key = key.key
        else:
            if self.varkeys:
                if key in self.varkeys:
                    keys = self.varkeys[key]
                    key = next(iter(keys))
                    if key.veckey:
                        key = key.veckey
                    elif len(keys) > 1:
                        raise ValueError("substitution key '%s' was ambiguous;"
                                         " .variables_byname('%s') will show"
                                         " which variables it may refer to."
                                         % (key, key))
                else:
                    raise KeyError("key '%s' does not refer to any varkey in"
                                   " this ConstraintSet" % key)
        idx = None
        if self.collapse_arrays:
            idx = getattr(key, "idx", None)
            if idx:
                key = key.veckey
        return key, idx

    def __contains__(self, key):
        "In a winding way, figures out if a key is in the KeyDict"
        key, idx = self.parse_and_index(key)
        if dict.__contains__(self, key):
            if idx:
                return not np.isnan(dict.__getitem__(self, key)[idx])
            return True
        elif key in self.keymap:
            return True
        else:
            return False

    def __getitem__(self, key):
        "Overloads __getitem__ and [] access to work with all keys"
        key, idx = self.parse_and_index(key)
        keys = self.keymap[key]
        if not keys:
            del self.keymap[key] # remove blank entry added due to defaultdict
            raise KeyError("%s was not found." % key)
        values = []
        for key in keys:
            got = dict.__getitem__(self, key)
            if idx:
                got = got[idx]
            values.append(got)
        if len(values) == 1:
            return values[0]
        else:
            return KeyDict(zip(keys, values))

    def __setitem__(self, key, value):
        "Overloads __setitem__ and []= to work with all keys"
        key, idx = self.parse_and_index(key)
        if key not in self.keymap:
            self.keymap[key].add(key)
            if hasattr(key, "keys"):
                for mapkey in key.keys:
                    self.keymap[mapkey].add(key)
            if idx:
                number_array = isinstance(value, Numbers)
                kwargs = {} if number_array else {"dtype": "object"}
                emptyvec = np.full(key.shape, np.nan, **kwargs)
                dict.__setitem__(self, key, emptyvec)
        for key in self.keymap[key]:
            if idx:
                dict.__getitem__(self, key)[idx] = value
            elif (dict.__contains__(self, key) and hasattr(value, "shape")
                  and np.isnan(value).any()):
                goodvals = ~np.isnan(value)
                self[key][goodvals] = value[goodvals]
            else:
                dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        "Overloads del [] to work with all keys"
        key, idx = self.parse_and_index(key)
        keys = self.keymap[key]
        if not keys:
            raise KeyError("key %s not found." % key)
        for key in list(keys):
            delete = True
            if idx:
                dict.__getitem__(self, key)[idx] = np.nan
                if np.isfinite(dict.__getitem__(self, key)).any():
                    delete = False
            if delete:
                dict.__delitem__(self, key)
                if hasattr(key, "keys"):
                    for mappedkey in key.keys:
                        self.keymap[mappedkey].remove(key)
                        if not self.keymap[mappedkey]:
                            del self.keymap[mappedkey]


class KeySet(KeyDict):
    "KeySets are KeyDicts without values, serving only to filter and map keys"
    collapse_arrays = False

    def add(self, item):
        "Adds an item to the keyset"
        self[item] = None

    def update(self, *args, **kwargs):
        "Iterates through the dictionary created by args and kwargs"
        if len(args) == 1:  # set-like interface
            for item in args[0]:
                self.add(item)
        else:  # dict-like interface
            for k in dict(*args, **kwargs):
                self.add(k)

    def __getitem__(self, key):
        "Gets the keys corresponding to a particular key."
        key, _ = self.parse_and_index(key)
        return self.keymap[key]

    def __setitem__(self, key, value):
        "Assigns the key itself every time."
        KeyDict.__setitem__(self, key, None)
