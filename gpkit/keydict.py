"Implements KeyDict and KeySet classes"
from collections import defaultdict, Hashable
import numpy as np
from .small_classes import Numbers, Quantity, FixedScalar
from .small_scripts import is_sweepvar, isnan, SweepValue

DIMLESS_QUANTITY = Quantity(1, "dimensionless")
INT_DTYPE = np.dtype(int)


def clean_value(key, value):
    """Gets the value of variable-less monomials, so that
    `x.sub({x: gpkit.units.m})` and `x.sub({x: gpkit.ureg.m})` are equivalent.

    Also converts any quantities to the key's units, because quantities
    can't/shouldn't be stored as elements of numpy arrays.
    """
    if hasattr(value, "__len__"):
        return [clean_value(key, v) for v in value]
    if isinstance(value, FixedScalar):
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
    keymap = []

    def __init__(self, *args, **kwargs):
        "Passes through to dict.__init__ via the `update()` method"
        # pylint: disable=super-init-not-called
        self.varkeys = None
        self.keymap = defaultdict(set)
        self._unmapped_keys = set()
        self.log_gets = False
        self.logged_gets = set()
        self.owned = set()
        self.update(*args, **kwargs)

    def get(self, key, alternative=KeyError):
        if key not in self:
            if alternative is KeyError:
                raise alternative(key)
            return alternative
        return self[key]

    def _copyonwrite(self, key):
        "Copys arrays before they are written to"
        if not hasattr(self, "owned"):  # backwards pickle compatibility
            self.owned = set()
        if key not in self.owned:
            dict.__setitem__(self, key, dict.__getitem__(self, key).copy())
            self.owned.add(key)

    def update(self, *args, **kwargs):
        "Iterates through the dictionary created by args and kwargs"
        if not self and len(args) == 1 and isinstance(args[0], KeyDict):
            dict.update(self, args[0])
            self.keymap.update(args[0].keymap)
            self._unmapped_keys.update(args[0]._unmapped_keys)  # pylint:disable=protected-access
        else:
            for k, v in dict(*args, **kwargs).items():
                self[k] = v

    def parse_and_index(self, key):
        "Returns key if key had one, and veckey/idx for indexed veckeys."
        try:
            key = key.key
            if self.collapse_arrays and key.idx:
                return key.veckey, key.idx
            return key, None
        except AttributeError:
            idx = None
            if not self.varkeys:
                self.update_keymap()
            elif key in self.varkeys:
                keys = self.varkeys[key]
                origkey, key = key, next(iter(keys))
                if len(keys) > 1:
                    if (key.veckey
                            and all(k.veckey == key.veckey for k in keys)):
                        key = key.veckey
                    else:
                        raise ValueError("%s could refer to multiple keys in"
                                         " this substitutions KeyDict. Use"
                                         " `.variables_byname(%s)` to see all"
                                         " of them." % (origkey, origkey))
            else:
                raise KeyError(key)
            if self.collapse_arrays:
                idx = getattr(key, "idx", None)
                if idx:
                    key = key.veckey
            return key, idx

    def __contains__(self, key):  # pylint:disable=too-many-return-statements
        "In a winding way, figures out if a key is in the KeyDict"
        try:
            key, idx = self.parse_and_index(key)
        except KeyError:
            return False
        except ValueError:  # multiple keys correspond
            return True
        if not isinstance(key, Hashable):
            return False
        if dict.__contains__(self, key):
            if idx:
                try:
                    value = dict.__getitem__(self, key)[idx]
                    return True if is_sweepvar(value) else not isnan(value)
                except TypeError:
                    raise TypeError("%s has an idx, but its value in this"
                                    " KeyDict is the scalar %s."
                                    % (key, dict.__getitem__(self, key)))
                except IndexError:
                    raise IndexError("key %s with idx %s is out of bounds"
                                     " for value %s" %
                                     (key, idx,
                                      dict.__getitem__(self, key)))
            return True
        elif key in self.keymap:
            return True
        else:
            return False

    def __call__(self, key):
        got = self[key]
        # if uniting ever becomes a speed hit, cache the results
        if isinstance(got, dict):
            for k, v in got.items():
                got[k] = v*(k.units or DIMLESS_QUANTITY)
        else:
            if not hasattr(key, "units"):
                parsedkey, _ = self.parse_and_index(key)
                keys = self.keymap[parsedkey]
                key, = keys
            got = Quantity(got, key.units or DIMLESS_QUANTITY)
        return got

    def __getitem__(self, key):
        "Overloads __getitem__ and [] access to work with all keys"
        key, idx = self.parse_and_index(key)
        keys = self.keymap[key]
        if not keys:
            del self.keymap[key]  # remove blank entry added due to defaultdict
            raise KeyError(key)
        values = []
        for k in keys:
            if not idx and k.shape:
                self._copyonwrite(k)
            got = dict.__getitem__(self, k)
            if idx:
                got = got[idx]
            values.append(got)
            if self.log_gets:
                self.logged_gets.add(k)
        if len(values) == 1:
            return values[0]
        return dict(zip(keys, values))

    def __setitem__(self, key, value):
        "Overloads __setitem__ and []= to work with all keys"
        # pylint: disable=too-many-boolean-expressions
        key, idx = self.parse_and_index(key)
        if key not in self.keymap:
            if not hasattr(self, "_unmapped_keys"):
                self.__init__()  # py3's pickle sets items before init... :(
            self.keymap[key].add(key)
            self._unmapped_keys.add(key)
            if idx:
                number_array = isinstance(value, Numbers)
                kwargs = {} if number_array else {"dtype": "object"}
                emptyvec = np.full(key.shape, np.nan, **kwargs)
                dict.__setitem__(self, key, emptyvec)
                self.owned.add(key)
        if isinstance(value, FixedScalar):
            value = value.value  # substitute constant monomials
        if isinstance(value, Quantity):
            value = value.to(key.units or "dimensionless").magnitude
        if idx:
            if is_sweepvar(value):
                dict.__setitem__(self, key,
                                 np.array(dict.__getitem__(self, key), object))
                self.owned.add(key)
                value = SweepValue(value[1])
            self._copyonwrite(key)
            dict.__getitem__(self, key)[idx] = value
        else:
            if (self.collapse_arrays and hasattr(key, "descr")
                    and "shape" in key.descr  # if veckey, and
                    and not isinstance(value, np.ndarray)  # not an array, and
                    and not is_sweepvar(value)):  # not a sweep, then
                if not hasattr(value, "__len__"):  # fill an array with it, or
                    value = np.full(key.shape, value, "f")
                # if it's not a list of arrays (as in a sol), clean it up!
                elif not isinstance(value[0], np.ndarray):
                    value = np.array([clean_value(key, v) for v in value])
            if getattr(value, "shape", False) and dict.__contains__(self, key):
                goodvals = ~isnan(value)
                present_value = dict.__getitem__(self, key)
                if present_value.dtype != value.dtype:
                    # e.g., we're replacing a number with a linked function
                    dict.__setitem__(self, key, np.array(present_value,
                                                         dtype=value.dtype))
                    self.owned.add(key)
                self._copyonwrite(key)
                self[key][goodvals] = value[goodvals]
            else:
                if hasattr(value, "dtype") and value.dtype == INT_DTYPE:
                    value = np.array(value, "f")
                dict.__setitem__(self, key, value)
                self.owned.add(key)

    def update_keymap(self):
        "Updates the keymap with the keys in _unmapped_keys"
        copied = set()  # have to copy bc update leaves duplicate sets
        while self.keymapping and self._unmapped_keys:
            key = self._unmapped_keys.pop()
            if hasattr(key, "keys"):
                for mapkey in key.keys:
                    if mapkey not in copied and mapkey in self.keymap:
                        self.keymap[mapkey] = set(self.keymap[mapkey])
                        copied.add(mapkey)
                    self.keymap[mapkey].add(key)

    def __delitem__(self, key):
        "Overloads del [] to work with all keys"
        key, idx = self.parse_and_index(key)
        keys = self.keymap[key]
        if not keys:
            raise KeyError(key)
        copied = set()  # have to copy bc update leaves duplicate sets
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
                for mapkey in mapkeys:
                    if mapkey in self.keymap:
                        if len(self.keymap[mapkey]) == 1:
                            del self.keymap[mapkey]
                            continue
                        if mapkey not in copied:
                            self.keymap[mapkey] = set(self.keymap[mapkey])
                            copied.add(mapkey)
                        self.keymap[mapkey].remove(k)


class KeySet(KeyDict):
    "KeyDicts that don't collapse arrays or store values."
    collapse_arrays = False

    def add(self, item):
        "Adds an item to the keyset"
        key, _ = self.parse_and_index(item)
        if key not in self.keymap:
            self.keymap[key].add(key)
            self._unmapped_keys.add(key)
            dict.__setitem__(self, key, None)

    def update(self, *args, **kwargs):
        "Iterates through the dictionary created by args and kwargs"
        if len(args) == 1:
            arg, = args
            if isinstance(arg, KeySet):  # assume unmapped
                dict.update(self, arg)
                for key, value in arg.keymap.items():
                    self.keymap[key].update(value)
                self._unmapped_keys.update(arg._unmapped_keys)  # pylint: disable=protected-access
            else:  # set-like interface
                for item in arg:
                    self.add(item)
        else:  # dict-like interface
            for k in dict(*args, **kwargs):
                self.add(k)

    def __getitem__(self, key):
        "Gets the keys corresponding to a particular key."
        key, _ = self.parse_and_index(key)
        return self.keymap[key]
