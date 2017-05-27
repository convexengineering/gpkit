from collections import defaultdict
import numpy as np
from .. import DimensionalityError
from ..small_classes import HashVector, Quantity, Strings
from ..keydict import KeySet
from ..small_scripts import mag
from ..varkey import VarKey
from .substitution import parse_subs


DIMLESS_QUANTITY = Quantity(1, "dimensionless")


class NomialMap(HashVector):

    def set_units(self, united_thing):
        self.units = None
        if hasattr(united_thing, "units"):
            self.units = Quantity(1, united_thing.units)
            try:  # faster than "if self.units.dimensionless"
                conversion = float(self.units)
                self.units = None
                for key, value in self.items():
                    self[key] = value*conversion
            except DimensionalityError:
                pass

    def to(self, units):
        sunits = self.units if self.units else DIMLESS_QUANTITY
        nm = self * sunits.to(units).magnitude
        nm.set_units(units)
        return nm

    def __add__(self, other):
        units = self.units
        if units != other.units and (units or other.units):
            if not units:
                unit_conversion = float(other.units)
                units = other.units
            else:
                unit_conversion = float(other.units/units)
            other = unit_conversion*other
        hmap = HashVector.__add__(self, other)
        hmap.set_units(units)
        hmap._remove_zeros(just_monomials=True)
        return hmap

    def _remove_zeros(self, just_monomials=False):
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
                    key._hashvalue = None  # reset hash
                    self[key] = value + self.get(key, 0)

    def diff(self, varkey):
        "Return differentiation of an hmap wrt a varkey"
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
        vunits = getattr(varkey, "units", None) or 1.0
        if isinstance(vunits, Strings):
            out.set_units(None)
        else:
            out.set_units((self.units or 1.0)/vunits)
        return out

    def sub(self, substitutions, varkeys, parsedsubs=False):
        if parsedsubs or not substitutions:
            fixed = substitutions
        else:
            fixed = parse_subs(varkeys, substitutions)

        if not substitutions:
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
                    expval, = cval.hmap.keys()
                    cval = cval.hmap
                    # TODO: can't-sub-posynomials error here
                if hasattr(cval, "to"):
                    if not vk.units or isinstance(vk.units, Strings):
                        vk.units = DIMLESS_QUANTITY
                    cval = mag(cval.to(vk.units))
                    if isinstance(cval, NomialMap) and cval.keys() == [{}]:
                        cval, = cval.values()
                if expval:
                    cval, = cval.values()
                exps_covered = set()
                for o_exp, exp in exps:
                    x = exp[vk]
                    # TODO: cval should already be a float
                    powval = float(cval)**x if cval != 0 or x >= 0 else np.inf
                    cp.csmap[o_exp] = powval * cp.csmap.get(o_exp, self[o_exp])
                    if exp in cp and exp not in exps_covered:
                        c = cp.pop(exp)
                        del exp[vk]
                        for key in expval:
                            exp[key] = expval[key]*x + exp.get(key, 0)
                        exp._hashvalue = None  # reset hash
                        cp[exp] = powval * c + cp.get(exp, 0)
                        exps_covered.add(exp)
        cp._remove_zeros(just_monomials=True)
        return cp

    def mmap(self, orig):
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
