"Implements KeyDict and KeySet classes"
from collections import defaultdict
import numpy as np
from .small_classes import Numbers


class KeyDict(dict):
    """KeyDicts do two things over a dict: map keys and collapse arrays.

    >>>> kd = gpkit.keydict.KeyDict()

    Mapping keys
    ------------
    If ``.keymapping`` is True, a KeyDict keeps an internal list of VarKeys as
    canonical keys, and their values can be accessed with any object whose
    `key` attribute matches one of those VarKeys, or with strings matching
    any of the multiple possible string interpretations of each key:

    For example, after creating the KeyDict kd and setting kd[x] = v (where x
    is a Variable or VarKey), v can be accessed with by the following keys:
     - x
     - x.key
     - x.name (a string)
     - "x_modelname" (x's name including modelname)

    Note that if a item is set using a key that does not have a `.key`
    attribute, that key can be set and accessed normally.

    Collapsing arrays
    -----------------
    If ``.collapse_arrays`` is True then VarKeys which have a `shape`
    parameter (indicating they are part of an array) are stored as numpy
    arrays, and automatically de-indexed when a matching VarKey with a
    particular `idx` parameter is used as a key.

    See also: gpkit/tests/t_keydict.py.
    """
    collapse_arrays = True
    keymapping = True

    def __init__(self, *args, **kwargs):
        "Passes through to dict.__init__ via the `update()` method"
        # pylint: disable=super-init-not-called
        self.varkeys = None
        self.keymap = defaultdict(set)
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        "Iterates through the dictionary created by args and kwargs"
        for k, v in dict(*args, **kwargs).items():
            if hasattr(v, "copy"):
                # We don't want just a reference (for e.g. numpy arrays)
                #   KeyDict values are expected to be immutable (Numbers)
                #   or to have a copy attribute.
                v = v.copy()
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
                try:
                    return not np.isnan(dict.__getitem__(self, key)[idx])
                except TypeError:
                    raise TypeError("%s has an idx, but its value in this"
                                    " KeyDict is the scalar %s."
                                    % (key, dict.__getitem__(self, key)))
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
            del self.keymap[key]  # remove blank entry added due to defaultdict
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
            if hasattr(key, "keys") and self.keymapping:
                for mapkey in key.keys:
                    self.keymap[mapkey].add(key)
            if idx:
                number_array = isinstance(value, Numbers)
                kwargs = {} if number_array else {"dtype": "object"}
                emptyvec = np.full(key.shape, np.nan, **kwargs)
                dict.__setitem__(self, key, emptyvec)
        for key in self.keymap[key]:
            if getattr(value, "exp", None) is not None and not value.exp:
                # get the value of variable-less monomials
                # so that `x.sub({x: gpkit.units.m})`
                # and `x.sub({x: gpkit.ureg.m})`
                # are equivalent
                value = value.value
            if idx:
                dict.__getitem__(self, key)[idx] = value
            else:
                if dict.__contains__(self, key) and getattr(value, "shape", ()):
                    try:
                        goodvals = ~np.isnan(value)
                    except TypeError:
                        pass  # could not evaluate nan-ness! assume no nans
                    else:
                        self[key][goodvals] = value[goodvals]
                        continue
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
                mapkeys = set([key])
                if self.keymapping and hasattr(key, "keys"):
                    mapkeys.update(key.keys)
                for mappedkey in mapkeys:
                    self.keymap[mappedkey].remove(key)
                    if not self.keymap[mappedkey]:
                        del self.keymap[mappedkey]


class KeySet(KeyDict):
    "KeyDicts that don't collapse arrays or store values."
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


class FastKeyDict(KeyDict):
    "KeyDicts that don't map keys, only collapse arrays"
    keymapping = False
