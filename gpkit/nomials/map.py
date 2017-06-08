"Implements the NomialMap class"
from collections import defaultdict
import numpy as np
from .. import DimensionalityError
from ..small_classes import HashVector, Quantity, Strings
from ..small_scripts import mag
from ..varkey import VarKey
from .substitution import parse_subs


DIMLESS_QUANTITY = Quantity(1, "dimensionless")


class NomialMap(HashVector):
    "Class for efficent algebraic represention of a nomial"
    units = None
    expmap = None
    csmap = None

    def set_units(self, thing, thing2=None):
        "Sets units to those of `thing*thing2`"
        if thing is None and thing2 is None:
            self.units = None
        elif hasattr(thing, "units"):
            if hasattr(thing2, "units"):
                self.units = Quantity(1, (thing*thing2).units)
                try:  # faster than "if self.units.dimensionless"
                    conversion = float(self.units)
                    self.units = None
                    for key, value in self.items():
                        self[key] = value*conversion
                except DimensionalityError:
                    pass
            else:
                self.units = Quantity(1, thing.units)
        elif hasattr(thing2, "units"):
            self.units = Quantity(1, thing2.units)
        else:
            self.units = None

    def to(self, units):
        "Converts a NomialMap to the given units"
        sunits = self.units if self.units else DIMLESS_QUANTITY
        nm = self * sunits.to(units).magnitude
        nm.set_units(units)  # pylint: disable=no-member
        return nm

    def __add__(self, other):
        "Adds NomialMaps together"
        if self.units != other.units:
            other = float(other.units/self.units)*other
        hmap = HashVector.__add__(self, other)
        hmap.units = self.units
        hmap.remove_zeros(just_monomials=True)   # pylint: disable=no-member, protected-access
        return hmap

    def remove_zeros(self, just_monomials=False):
        "Removes zeroed exponents and monomials."
        # TODO: do this automatically during HashVector operations?
        posynomial = (len(self) > 1)
        for key in self.keys():
            value = self[key]
            if posynomial and value == 0:
                del self[key]
            elif not just_monomials:
                zeroes = set(vk for vk, exp in key.items() if exp == 0)
                if zeroes:
                    del self[key]
                    for vk in zeroes:
                        del key[vk]
                    key._hashvalue = None  # reset hash # pylint: disable=protected-access
                    self[key] = value + self.get(key, 0)

    def diff(self, varkey):
        "Differentiates a NomialMap with respect to a varkey"
        out = NomialMap()
        for exp in self:
            if varkey in exp:
                exp = HashVector(exp)
                x = exp[varkey]
                c = self[exp] * x
                if x is 1:
                    del exp[varkey]
                else:
                    exp[varkey] = x-1
                out[exp] = c
        out.set_units(self.units, 1.0/varkey.units if varkey.units else None)
        return out

    def sub(self, substitutions, varkeys, parsedsubs=False):
        "Applies substitutions to a NomialMap"
        # pylint: disable=too-many-locals, too-many-branches
        if parsedsubs or not substitutions:
            fixed = substitutions
        else:
            fixed = parse_subs(varkeys, substitutions)

        if not substitutions:
            if not self.expmap:
                self.expmap, self.csmap = {exp: exp for exp in self}, {}
            return self

        cp = NomialMap()
        cp.units = self.units
        cp.expmap, cp.csmap = {}, {}
        varlocs = defaultdict(set)
        for exp, c in self.items():
            new_exp = HashVector(exp)
            cp.expmap[exp] = new_exp
            cp[new_exp] = c
            for vk in new_exp:
                varlocs[vk].add((exp, new_exp))

        for vk in varlocs:
            if vk in fixed:
                expval = []
                exps, cval = varlocs[vk], fixed[vk]
                if isinstance(cval, Strings):
                    descr = dict(vk.descr)
                    del descr["name"]
                    cval = VarKey(name=cval, **descr)
                if hasattr(cval, "hmap"):
                    expval, = cval.hmap.keys()  # TODO: catch "can't-sub-posys"
                    cval = cval.hmap  # pylint: disable=redefined-variable-type
                if hasattr(cval, "to"):
                    cval = mag(cval.to(vk.units or DIMLESS_QUANTITY))
                    if isinstance(cval, NomialMap) and cval.keys() == [{}]:
                        cval, = cval.values()
                if expval:
                    cval, = cval.values()
                exps_covered = set()
                for o_exp, exp in exps:
                    x = exp[vk]
                    powval = float(cval)**x if cval != 0 or x >= 0 else np.inf
                    cp.csmap[o_exp] = powval * cp.csmap.get(o_exp, self[o_exp])
                    if exp in cp and exp not in exps_covered:
                        c = cp.pop(exp)
                        del exp[vk]
                        for key in expval:
                            exp[key] = expval[key]*x + exp.get(key, 0)
                        exp._hashvalue = None  # reset hash, # pylint: disable=protected-access
                        cp[exp] = powval * c + cp.get(exp, 0)
                        exps_covered.add(exp)
                        cp.remove_zeros(just_monomials=True)  # pylint: disable=protected-access
        return cp

    def mmap(self, orig):
        "Maps substituted monomials back to the original nomial"
        m_from_ms = defaultdict(dict)
        pmap = [{} for _ in self]
        origexps = orig.keys()
        selfexps = self.keys()
        for orig_exp, self_exp in self.expmap.items():
            total_c = self.get(self_exp, None)
            if total_c:
                fraction = self.csmap.get(orig_exp, orig[orig_exp])/total_c
                m_from_ms[self_exp][orig_exp] = fraction
                orig_idx = origexps.index(orig_exp)
                pmap[selfexps.index(self_exp)][orig_idx] = fraction
        return pmap, m_from_ms
