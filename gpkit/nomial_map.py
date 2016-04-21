import numpy as np
from collections import defaultdict
from .small_classes import HashVector, Numbers, Quantity
from . import units as ureg
from .small_scripts import mag


class NomialMap(HashVector):

    def __init__(self, *args, **kwargs):
        super(NomialMap, self).__init__(*args, **kwargs)
        self.varlocs = defaultdict(set)
        for exp in self:
            for vk in exp:
                self.varlocs[vk].add(exp)
        # self._simplify()

    def set_units(self, united_thing):
        if hasattr(united_thing, "units"):
            self.units = Quantity(1, united_thing.units)
        else:
            self.units = None

    def _simplify(self, just_zero_monomials=False):
        posynomial = (len(self) > 1)
        for key in self.keys():
            value = self[key]
            if posynomial and value == 0:
                del self[key]
            elif not just_zero_monomials:
                zeroes = set(vk for vk, exp in key.items() if exp == 0)
                if zeroes:
                    del self[key]
                    for vk in zeroes:
                        del key[vk]
                    key._hashvalue = None
                    self[key] = value + self.get(key, 0)

    def to(self, units):
        sunits = self.units if self.units else Quantity(1, ureg.dimensionless)
        nm = self*sunits.to(units).magnitude
        nm.units = units
        return nm

    def sub(self, substitutions):
        exps_touched = set()
        for vk, exps in self.varlocs.items():
            if vk in substitutions:
                value = substitutions[vk]
                if hasattr(value, "key"):
                    self.hmap = NomialMap({HashVector({value.key: 1}): 1.0})
                    self.hmap.units = value.key.units if ureg else None
                if hasattr(value, "hmap"):
                    assert len(value.hmap) == 1
                    value = value.hmap
                    monomial_sub = True
                if hasattr(value, "to"):
                    if not vk.units:
                        self.units = "dimensionless"
                    value = mag(value.to(vk.units))
                for exp in exps:
                    x = exp.pop(vk)
                    if monomial_sub:
                        m_c, = value.values()
                        m_exp, = value.keys()
                        old_c = self.pop(exp)
                        exp += m_exp*x
                        self[exp] = old_c * m_c**x
                    else:
                        self[exp] *= value**x
                    exps_touched.add(exp)
        for exp in exps_touched:
            # pmap here
            value = self.pop(exp)
            exp._hashvalue = None
            self[exp] = value + self.get(exp, 0)

    def __add__(self, other):
        units = self.units
        unit_conversion = None
        if not (self.units or other.units):
            pass
        elif not self.units:
            unit_conversion = other.units.to(ureg.dimensionless)
            units = other.units
        elif not other.units:
            unit_conversion = self.units.to(ureg.dimensionless)
        elif self.units != other.units:
            unit_conversion = (other.units/self.units).to(ureg.dimensionless)
        if unit_conversion:
            other = unit_conversion*other

        hmap = HashVector.__add__(self, other)
        hmap.set_units(units)
        hmap._simplify(just_zero_monomials=True)
        return hmap

    @classmethod
    def from_exps_and_cs(cls, exps, cs):
        if isinstance(cs, Quantity):
            units = Quantity(1, cs.units)
            cs = cs.magnitude
        else:
            units = None
        nm = cls(zip(exps, cs))
        nm.set_units(units)
        return nm

    @classmethod
    def simplify_exps_and_cs(cls, exps, cs):
        if isinstance(cs, Quantity):
            units = Quantity(1, cs.units)
            cs = cs.magnitude
        elif isinstance(cs[0], Quantity):
            units = Quantity(1, cs[0].units)
            if len(cs) == 1:
                cs = [cs[0].magnitude]
            else:
                cs = [c.to(units).magnitude for c in cs]
        else:
            units = None

        matches = defaultdict(float)
        for i, exp in enumerate(exps):
            exp = HashVector(exp)
            matches[exp] += cs[i]

        nm = cls(matches)
        nm.set_units(units)
        nm._simplify()
        exps_ = tuple(nm.keys())
        cs_ = list(nm.values())

        cs_ = np.array(nm.values())
        if units:
            cs_ = cs_*units

        return nm, exps_, cs_
