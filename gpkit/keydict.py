import numpy as np
from collections import defaultdict
from .small_classes import Numbers, Strings


class KeyDict(dict):
    collapse_arrays = True

    def __init__(self, *args, **kwargs):
        self.baked_keystrs = None
        self.update(*args, **kwargs)

    @classmethod
    def with_keys(cls, varkeys, *dictionaries):
        out = cls()
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                keys = varkeys[key]
                for key in keys:
                    if not key.idx:
                        out[key] = value
                    else:
                        if not hasattr(value, "shape"):
                            value = np.array(value)
                        if not np.isnan(value[key.idx]):
                            out[key] = value[key.idx]
        return out

    @classmethod
    def from_constraints(cls, varkeys, constraints, substitutions=None):
        substitutions = substitutions if substitutions else {}
        constraintsubs = []
        for constraint in constraints:
            constraintsubs.append(constraint.substitutions)
            constraint.substitutions = {}
        sublist = constraintsubs + [substitutions]
        return cls.with_keys(varkeys, *sublist)

    def __contains__(self, key):
        if dict.__contains__(self, key):
            return True
        elif hasattr(key, "key"):
            if dict.__contains__(self, key.key):
                return True
            elif self.is_veckey_but_not_collapsed(key):
                if any(k in self for k in self.getkeys(key)):
                    return True
            elif self.is_index_into_vector(key.key):
                if dict.__contains__(self, veckeyed(key.key)):
                    return True
        elif key in self.keystrs():
            return True

        return False

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def bake(self):
        self.baked_keystrs = None
        self.baked_keystrs = self.keystrs()

    def keystrs(self):
        if self.baked_keystrs:
            return self.baked_keystrs
        keystrs = defaultdict(set)
        for key in self.keys():
            for keystr in key.allstrs:
                keystrs[keystr].add(key)
        return keystrs

    def getkeys(self, key):
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
        if "shape" not in key.descr:
            return False
        if "idx" in key.descr:
            return False
        if self.collapse_arrays:
            return False
        return True

    def is_index_into_vector(self, key):
        if not self.collapse_arrays:
            return False
        if "idx" not in key.descr:
            return False
        if "shape" not in key.descr:
            return False
        return True

    def __dgi(self, key):
        return dict.__getitem__(self, key)

    def __getitem__(self, key):
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
        for key in self.getkeys(key):
            if key in self:
                dict.__delitem__(self, key)
            elif "shape" in key.descr and "idx" in key.descr:
                self.__dgi(veckeyed(key))[key.descr["idx"]] = np.nan


class KeySet(KeyDict):
    collapse_arrays = False

    def __getitem__(self, key):
        if key not in self:
            return []
        return [k for k in self.getkeys(key) if k in self]

    def __setitem__(self, key, value):
        KeyDict.__setitem__(self, key, None)

    def map(self, iterable):
        varkeys = []
        for key in iterable:
            keys = self[key]
            if len(keys) > 1:
                raise ValueError("KeySet.map() only accepts unambiguous keys.")
            key, = keys
            varkeys.append(key)
        if len(varkeys) == 1:
            varkeys = varkeys[0]
        return varkeys


def veckeyed(key):
    vecdescr = dict(key.descr)
    del vecdescr["idx"]
    vecdescr.pop("value", None)
    return key.__class__(**vecdescr)
