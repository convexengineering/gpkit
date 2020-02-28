"Implements the NomialMap class"
from collections import defaultdict
import numpy as np
from .. import units
from ..exceptions import DimensionalityError
from ..small_classes import HashVector, Strings, qty, EMPTY_HV
from .substitution import parse_subs

DIMLESS_QUANTITY = qty("dimensionless")


class NomialMap(HashVector):
    """Class for efficent algebraic represention of a nomial

    A NomialMap is a mapping between hashvectors representing exponents
    and their coefficients in a posynomial.

    For example,  {{x : 1}: 2.0, {y: 1}: 3.0} represents 2*x + 3*y, where
    x and y are VarKey objects.
    """
    units = None
    expmap = None  # used for monomial-mapping postsubstitution; see .mmap()
    csmap = None   # used for monomial-mapping postsubstitution; see .mmap()

    def copy(self):
        "Return a copy of this"
        return self.__class__(self)

    def units_of_product(self, thing, thing2=None):
        "Sets units to those of `thing*thing2`. Ugly optimized code."
        if thing is None and thing2 is None:
            self.units = None
        elif hasattr(thing, "units"):
            if hasattr(thing2, "units"):
                self.units, dimless_convert = units.of_product(thing, thing2)
                if dimless_convert:
                    for key in self:
                        self[key] *= dimless_convert
            else:
                self.units = qty(thing.units)
        elif hasattr(thing2, "units"):
            self.units = qty(thing2.units)
        elif thing2 is None and isinstance(thing, Strings):
            self.units = qty(thing)
        else:
            self.units = None

    def to(self, to_units):
        "Returns a new NomialMap of the given units"
        sunits = self.units or DIMLESS_QUANTITY
        nm = self * sunits.to(to_units).magnitude  # note that * creates a copy
        nm.units_of_product(to_units)  # pylint: disable=no-member
        return nm

    def __add__(self, other):
        "Adds NomialMaps together"
        if self.units != other.units:
            try:
                other *= float(other.units/self.units)
            except (TypeError, AttributeError):  # if one of those is None
                raise DimensionalityError(self.units, other.units)
        hmap = HashVector.__add__(self, other)
        hmap.units = self.units
        return hmap

    def diff(self, varkey):
        "Differentiates a NomialMap with respect to a varkey"
        out = NomialMap()
        out.units_of_product(self.units,
                             1/varkey.units if varkey.units else None)
        for exp in self:
            if varkey in exp:
                x = exp[varkey]
                c = self[exp] * x
                exp = exp.copy()
                if x == 1:
                    exp.hashvalue ^= hash((varkey, 1))
                    del exp[varkey]
                else:
                    exp.hashvalue ^= hash((varkey, x)) ^ hash((varkey, x-1))
                    exp[varkey] = x-1
                out[exp] = c
        return out

    def sub(self, substitutions, varkeys, parsedsubs=False):
        """Applies substitutions to a NomialMap

        Parameters
        ----------
        substitutions : (dict-like)
            list of substitutions to perform

        varkeys : (set-like)
            varkeys that are present in self
            (required argument so as to require efficient code)

        parsedsubs : bool
            flag if the substitutions have already been parsed
            to contain only keys in varkeys

        """
        # pylint: disable=too-many-locals, too-many-branches
        if parsedsubs or not substitutions:
            fixed = substitutions
        else:
            fixed, _, _ = parse_subs(varkeys, substitutions)

        if not fixed:
            if not self.expmap:
                self.expmap, self.csmap = {exp: exp for exp in self}, {}
            return self

        cp = NomialMap()
        cp.units = self.units
        # csmap is modified during substitution, but keeps the same exps
        cp.expmap, cp.csmap = {}, self.copy()
        varlocs = defaultdict(set)
        for exp, c in self.items():
            new_exp = exp.copy()
            cp.expmap[exp] = new_exp  # cp modifies exps, so it needs new ones
            cp[new_exp] = c
            for vk in new_exp:
                if vk in fixed:
                    varlocs[vk].add((exp, new_exp))

        squished = set()
        for vk in varlocs:
            exps, cval = varlocs[vk], fixed[vk]
            if hasattr(cval, "hmap"):
                if any(cval.hmap.keys()):
                    raise ValueError("Monomial substitutions are not"
                                     " supported.")
                cval, = cval.hmap.to(vk.units or DIMLESS_QUANTITY).values()
            elif hasattr(cval, "to"):
                cval = cval.to(vk.units or DIMLESS_QUANTITY).magnitude
            for o_exp, exp in exps:
                subinplace(cp, exp, o_exp, vk, cval, squished)
        return cp

    def mmap(self, orig):
        """Maps substituted monomials back to the original nomial

        self.expmap is the map from pre- to post-substitution exponents, and
            takes the form {original_exp: new_exp}

        self.csmap is the map from pre-substitution exponents to coefficients.

        m_from_ms is of the form {new_exp: [old_exps, ]}

        pmap is of the form [{orig_idx1: fraction1, orig_idx2: fraction2, }, ]
            where at the index corresponding to each new_exp is a dictionary
            mapping the indices corresponding to the old exps to their
            fraction of the post-substitution coefficient
        """
        m_from_ms = defaultdict(dict)
        pmap = [{} for _ in self]
        origexps = list(orig.keys())
        selfexps = list(self.keys())
        for orig_exp, self_exp in self.expmap.items():
            total_c = self.get(self_exp, None)  # TODO: seems unnecessary?
            if total_c:
                fraction = self.csmap.get(orig_exp, orig[orig_exp])/total_c
                m_from_ms[self_exp][orig_exp] = fraction
                orig_idx = origexps.index(orig_exp)
                pmap[selfexps.index(self_exp)][orig_idx] = fraction
        return pmap, m_from_ms


# pylint: disable=invalid-name
def subinplace(cp, exp, o_exp, vk, cval, squished):
    "Modifies cp by substituing cval/expval for vk in exp"
    x = exp[vk]
    powval = float(cval)**x if cval != 0 or x >= 0 else np.sign(cval)*np.inf
    cp.csmap[o_exp] *= powval
    if exp in cp:
        c = cp.pop(exp)
        exp.hashvalue ^= hash((vk, x))  # remove (key, value) from hashvalue
        del exp[vk]
        value = powval * c
        if exp in cp:
            squished.add(exp.copy())
            currentvalue = cp[exp]
            if value != -currentvalue:
                cp[exp] += value
            else:
                del cp[exp]  # remove zeros created during substitution
        elif value:
            cp[exp] = value
        if not cp:  # make sure it's never an empty hmap
            cp[EMPTY_HV] = 0.0
    elif exp in squished:
        exp.hashvalue ^= hash((vk, x))  # remove (key, value) from hashvalue
        del exp[vk]
