"Implements the NomialMap class"
from collections import defaultdict
import numpy as np
from ..exceptions import DimensionalityError
from ..small_classes import HashVector, Quantity, Strings, qty
from ..small_scripts import mag
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

    def units_of_product(self, thing, thing2=None):
        "Sets units to those of `thing*thing2`"
        if thing is None and thing2 is None:
            self.units = None
        elif hasattr(thing, "units"):
            if hasattr(thing2, "units"):
                self.units = qty((thing*thing2).units)
                try:  # faster than "if self.units.dimensionless"
                    conversion = float(self.units)
                    self.units = None
                    for key in self:
                        self[key] *= conversion
                except DimensionalityError:
                    pass
            elif not isinstance(thing, Quantity):
                self.units = thing.units
            else:
                self.units = qty(thing.units)
        elif hasattr(thing2, "units"):
            self.units = qty(thing2.units)
        elif thing2 is None and isinstance(thing, Strings):
            self.units = qty(thing)
        else:
            self.units = None

    def to(self, units):
        "Returns a new NomialMap of the given units"
        sunits = self.units or DIMLESS_QUANTITY
        nm = self * sunits.to(units).magnitude  # note that * creates a copy
        nm.units_of_product(units)  # pylint: disable=no-member
        return nm

    def __add__(self, other):
        "Adds NomialMaps together"
        if self.units != other.units:
            try:
                other *= float(other.units/self.units)
            except TypeError:  # if one of those units is None
                raise DimensionalityError(self.units, other.units)
        hmap = HashVector.__add__(self, other)
        hmap.units = self.units
        return hmap

    def remove_zeros(self):
        """Removes zeroed exponents and monomials.

        If `only_check_cs` is True, checks only whether any values are zero.
        If False also checks whether any exponents in the keys are zero.
        """
        for key, value in self.items():
            zeroes = set(vk for vk, exp in key.items() if exp == 0)
            if zeroes:
                # raise ValueError(self)
                del self[key]
                for vk in zeroes:
                    key._hashvalue ^= hash((vk, key[vk]))
                    del key[vk]
                self[key] = value + self.get(key, 0)

    def diff(self, varkey):
        "Differentiates a NomialMap with respect to a varkey"
        out = NomialMap()
        for exp in self:
            if varkey in exp:
                exp = HashVector(exp)
                x = exp[varkey]
                c = self[exp] * x
                if x is 1:  # speed optimization
                    del exp[varkey]
                else:
                    exp[varkey] = x-1
                out[exp] = c
        out.units_of_product(self.units,
                             1.0/varkey.units if varkey.units else None)
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

        if not substitutions:
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
                varlocs[vk].add((exp, new_exp))

        for vk in varlocs:
            if vk in fixed:
                expval = []
                exps, cval = varlocs[vk], fixed[vk]
                if hasattr(cval, "hmap"):
                    expval, = cval.hmap.keys()  # TODO: catch "can't-sub-posys"
                    cval = cval.hmap
                if hasattr(cval, "to"):
                    cval = mag(cval.to(vk.units or DIMLESS_QUANTITY))
                    if isinstance(cval, NomialMap) and cval.keys() == [{}]:
                        cval, = cval.values()
                if expval:
                    cval, = cval.values()
                exps_covered = set()
                for o_exp, exp in exps:
                    subinplace(cp, exp, o_exp, vk, cval, expval, exps_covered)
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
            total_c = self.get(self_exp, None)
            if total_c:
                fraction = self.csmap.get(orig_exp, orig[orig_exp])/total_c
                m_from_ms[self_exp][orig_exp] = fraction
                orig_idx = origexps.index(orig_exp)
                pmap[selfexps.index(self_exp)][orig_idx] = fraction
        return pmap, m_from_ms


# pylint: disable=invalid-name
def subinplace(cp, exp, o_exp, vk, cval, expval, exps_covered):
    "Modifies cp by substituing cval/expval for vk in exp"
    x = exp[vk]
    powval = float(cval)**x if cval != 0 or x >= 0 else np.inf
    cp.csmap[o_exp] *= powval
    if exp in cp and exp not in exps_covered:
        c = cp.pop(exp)
        exp._hashvalue ^= hash((vk, x))  # remove (key, value) from _hashvalue
        del exp[vk]
        for key in expval:
            if key in exp:
                exp._hashvalue ^= hash((key, exp[key]))  # remove from hash
                newval = expval[key]*x + exp[key]
            else:
                newval = expval[key]*x
            exp._hashvalue ^= hash((key, newval))  # add to hash
            exp[key] = newval
        value = powval * c
        if exp in cp:
            currentvalue = cp[exp]
            if value != -currentvalue:
                cp[exp] = value + currentvalue
            else:
                del cp[exp]  # remove zeros created during substitution
        elif value:
            cp[exp] = value
        if not cp:  # make sure it's never an empty hmap
            cp[HashVector()] = 0.0
        exps_covered.add(exp)
