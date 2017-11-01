"Implements KeyDict and KeySet classes"
from collections import defaultdict
import numpy as np
from .small_classes import Numbers, Quantity
from .small_scripts import is_sweepvar


@np.vectorize
def isnan(element):
    "Determine if something of arbitrary type is a numpy nan."
    try:
        return np.isnan(element)
    except TypeError:
        return False


def clean_value(key, value):
    """Gets the value of variable-less monomials, so that
    `x.sub({x: gpkit.units.m})` and `x.sub({x: gpkit.ureg.m})` are equivalent.

    Also converts any quantities to the key's units, because quantities
    can't/shouldn't be stored as elements of numpy arrays.
    """
    if hasattr(value, "exp") and not value.exp:
        value = value.value
    if hasattr(value, "units") and not hasattr(value, "hmap"):
        value = value.to(key.units or "dimensionless").magnitude
    return value


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
        self._unmapped_keys = set()
        self.update(*args, **kwargs)

    def get(self, key, alternative=KeyError):
        if key not in self:
            if alternative is KeyError:
                raise alternative(key)
            return alternative
        return self[key]

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
        elif not self.varkeys:
            self.update_keymap()
        elif key in self.varkeys:
            keys = self.varkeys[key]
            origkey, key = key, next(iter(keys))
            if len(keys) > 1:
                if key.veckey and all(k.veckey == key.veckey for k in keys):
                    key = key.veckey
                else:
                    raise ValueError("%s could refer to multiple keys in this"
                                     " substitutions KeyDict. Use"
                                     " `.variables_byname(%s)` to see all of"
                                     " them." % (origkey, origkey))
        else:
            raise KeyError(key)
        idx = getattr(key, "idx", None)
        if not self.collapse_arrays:
            idx = None
        elif idx:
            key = key.veckey
        return key, idx

    def __contains__(self, key):
        "In a winding way, figures out if a key is in the KeyDict"
        try:
            key, idx = self.parse_and_index(key)
        except KeyError:
            return False
        except ValueError:  # multiple keys correspond
            return True
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
            raise KeyError(key)
        values = []
        for k in keys:
            got = dict.__getitem__(self, k)
            if idx:
                got = got[idx]
            values.append(got)
        if len(values) == 1:
            return values[0]
        return KeyDict(zip(keys, values))

    def __setitem__(self, key, value):
        "Overloads __setitem__ and []= to work with all keys"
        # pylint: disable=too-many-boolean-expressions
        key, idx = self.parse_and_index(key)
        if key not in self.keymap:
            self.keymap[key].add(key)
            self._unmapped_keys.add(key)
            if idx:
                number_array = isinstance(value, Numbers)
                kwargs = {} if number_array else {"dtype": "object"}
                emptyvec = np.full(key.shape, np.nan, **kwargs)
                dict.__setitem__(self, key, emptyvec)
        if hasattr(value, "exp") and not value.exp:
            value = value.value  # substitute constant monomials
        if idx:
            if isinstance(value, Quantity):
                value = value.to(key.units).magnitude
            dict.__getitem__(self, key)[idx] = value
        else:
            if (self.collapse_arrays and hasattr(key, "descr")
                    and "shape" in key.descr  # if veckey, not
                    and not isinstance(value, (np.ndarray, Quantity))  # array,
                    and not is_sweepvar(value)  # not sweep, and
                    and not isinstance(value[0], np.ndarray)):  # not solarray
                value = np.array([clean_value(key, v) for v in value])
            if getattr(value, "shape", False) and dict.__contains__(self, key):
                goodvals = ~isnan(value)
                self[key][goodvals] = value[goodvals]
            else:
                dict.__setitem__(self, key, value)

    def update_keymap(self):
        "Updates the keymap with the keys in _unmapped_keys"
        while self.keymapping and self._unmapped_keys:
            key = self._unmapped_keys.pop()
            if hasattr(key, "keys"):
                for mapkey in key.keys:
                    self.keymap[mapkey].add(key)

    def __delitem__(self, key):
        "Overloads del [] to work with all keys"
        key, idx = self.parse_and_index(key)
        keys = self.keymap[key]
        if not keys:
            raise KeyError("key %s not found." % key)
        for k in list(keys):
            delete = True
            if idx:
                dict.__getitem__(self, k)[idx] = np.nan
                if not isnan(dict.__getitem__(self, k)).all():
                    delete = False
            if delete:
                dict.__delitem__(self, k)
                mapkeys = set([k])
                if self.keymapping and hasattr(k, "keys"):
                    mapkeys.update(k.keys)
                for mappedkey in mapkeys:
                    if mappedkey in self.keymap:
                        self.keymap[mappedkey].remove(k)
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
