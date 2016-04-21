import numpy as np
from collections import defaultdict, Iterable
from .small_classes import HashVector, Numbers, Quantity, Strings
from .keydict import KeySet
from . import units as ureg
from .small_scripts import mag, is_sweepvar
from .varkey import VarKey


class NomialMap(HashVector):

    def set_units(self, united_thing):
        if hasattr(united_thing, "units"):
            self.units = Quantity(1, united_thing.units)
        else:
            self.units = None

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
                    key._hashvalue = None
                    self[key] = value + self.get(key, 0)

    def to(self, units):
        sunits = self.units if self.units else Quantity(1, ureg.dimensionless)
        nm = self*sunits.to(units).magnitude
        nm.units = units
        return nm

    def sub(self, substitutions, val=None, mmap=False):
        if val is not None:
            substitutions = {substitutions: val}

        cp = NomialMap()
        cp.units = self.units
        expmap = {}
        varlocs = defaultdict(set)
        for exp, c in self.items():
            new_exp = HashVector(exp)
            expmap[exp] = new_exp
            cp[new_exp] = c
            for vk in new_exp:
                varlocs[vk].add((exp, new_exp))

        varkeys = KeySet(varlocs)
        substitutions, _, _ = parse_subs(varkeys, substitutions)

        cs_map = {}
        for vk, exps in varlocs.items():
            if vk in substitutions:
                m_exp = []
                value = substitutions[vk]
                if isinstance(value, Strings):
                    descr = dict(vk.descr)
                    del descr["name"]
                    value = VarKey(name=value, **descr)
                if hasattr(value, "hmap"):
                    m_exp, = value.hmap.keys()
                    value = value.hmap
                    # TODO: can't-sub-posynomials error here
                if hasattr(value, "to"):
                    if not vk.units or isinstance(vk.units, Strings):
                        vk.units = ureg.dimensionless
                    value = mag(value.to(vk.units))
                if m_exp:
                    value, = value.values()
                for o_exp, exp in exps:
                    x = exp.pop(vk)
                    powval = value**x if value != 0 or x >= 0 else np.inf
                    cs_map[o_exp] = cs_map.get(o_exp, self[o_exp]) * powval
                    if exp in cp:
                        c = cp.pop(exp)
                        for key in m_exp:
                            exp[key] = m_exp[key]*x + exp.get(key, 0)
                        exp._hashvalue = None
                        cp[exp] = c * powval + cp.get(exp, 0)
                    else:
                        exp._hashvalue = None

        cp._remove_zeros()

        if not mmap:
            return cp

        m_from_ms = defaultdict(dict)
        pmap = [{} for _ in cp]
        selfexps = self.keys()
        cpexps = cp.keys()
        for self_exp, cp_exp in expmap.items():
            total_c = cp.get(cp_exp, None)
            if total_c:
                fraction = cs_map.get(self_exp, self[self_exp])/total_c
                m_from_ms[cp_exp][self_exp] = fraction
                pmap[cpexps.index(cp_exp)][selfexps.index(self_exp)] = fraction

        return cp, pmap, m_from_ms

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
        hmap._remove_zeros(just_monomials=True)
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
        nm._remove_zeros()
        exps_ = tuple(nm.keys())
        cs_ = list(nm.values())

        cs_ = np.array(nm.values())
        if units:
            cs_ = cs_*units

        return nm, exps_, cs_


def parse_subs(varkeys, substitutions):
    "Seperates subs into constants, sweeps linkedsweeps actually present."
    constants, sweep, linkedsweep = {}, {}, {}
    if hasattr(substitutions, "keymap"):
        for var in varkeys.keymap:
            if dict.__contains__(substitutions, var):
                sub = dict.__getitem__(substitutions, var)
                keys = varkeys.keymap[var]
                append_sub(sub, keys, constants, sweep, linkedsweep)
    else:
        for var in substitutions:
            key = getattr(var, "key", var)
            if key in varkeys.keymap:
                sub, keys = substitutions[var], varkeys.keymap[key]
                append_sub(sub, keys, constants, sweep, linkedsweep)
    return constants, sweep, linkedsweep


def append_sub(sub, keys, constants, sweep, linkedsweep):
    "Appends sub to constants, sweep, or linkedsweep."
    sweepsub = is_sweepvar(sub)
    if sweepsub:
        _, sub = sub  # _ catches the "sweep" marker
    for key in keys:
        if not key.shape or not isinstance(sub, Iterable):
            value = sub
        else:
            sub = np.array(sub) if not hasattr(sub, "shape") else sub
            if key.shape == sub.shape:
                value = sub[key.idx]
                if is_sweepvar(value):
                    _, value = value
                    sweepsub = True
            elif sweepsub:
                try:
                    np.broadcast(sub, np.empty(key.shape))
                except ValueError:
                    raise ValueError("cannot sweep variable %s of shape %s"
                                     " with array of shape %s; array shape"
                                     " must either be %s or %s" %
                                     (key.str_without("model"), key.shape,
                                      sub.shape,
                                      key.shape, ("N",)+key.shape))
                idx = (slice(None),)+key.descr["idx"]
                value = sub[idx]
            else:
                raise ValueError("cannot substitute array of shape %s for"
                                 " variable %s of shape %s." %
                                 (sub.shape, key.str_without("model"),
                                  key.shape))
        if not sweepsub:
            try:
                assert np.isnan(value)
            except (AssertionError, TypeError, ValueError):
                constants[key] = value
        elif not hasattr(value, "__call__"):
            sweep[key] = value
        else:
            linkedsweep[key] = value
