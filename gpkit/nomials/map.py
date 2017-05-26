import numpy as np
from collections import defaultdict, Iterable
from ..small_classes import HashVector, Quantity, Strings
from ..keydict import KeySet
from ..small_scripts import mag, is_sweepvar
from ..varkey import VarKey


DIMLESS_QUANTITY = Quantity(1, "dimensionless")


class NomialMap(HashVector):

    def set_units(self, united_thing):
        if hasattr(united_thing, "units"):
            self.units = Quantity(1, united_thing.units)
            if self.units.dimensionless:
                conversion = float(self.units)
                self.units = None
                for key, value in self.items():
                    self[key] = value*conversion
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
        sunits = self.units if self.units else DIMLESS_QUANTITY
        nm = self * sunits.to(units).magnitude
        nm.units = units
        return nm

    def diff(self, varkey):
        "Return differentiation of an hmap wrt a varkey"
        out = NomialMap()
        for exp in self:
            if varkey in exp:
                exp = HashVector(exp)
                c = self[exp] * exp[varkey]
                exp[varkey] -= 1
                out[exp] = c
        out._remove_zeros()
        vunits = getattr(varkey, "units", None) or 1.0
        if isinstance(vunits, Strings):
            out.set_units(None)
        else:
            sunits = self.units or 1.0
            out.set_units(sunits/vunits)
        return out

    def sub(self, substitutions, val=None):
        if val is not None:
            substitutions = {substitutions: val}

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

        if not substitutions:
            return cp
        fixed = parse_subs(KeySet(varlocs), substitutions, sweeps=False)
        if not fixed:
            return cp

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
                        exp._hashvalue = None
                        cp[exp] = powval * c + cp.get(exp, 0)
                        exps_covered.add(exp)
                    else:
                        exp._hashvalue = None
        cp._remove_zeros()
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

    def __add__(self, other):
        units = self.units
        unit_conversion = None
        if not (self.units or other.units):
            pass
        elif not self.units:
            unit_conversion = other.units.to("dimensionless")
            units = other.units
        elif not other.units:
            unit_conversion = 1.0/self.units.to("dimensionless")
        elif self.units != other.units:
            unit_conversion = (other.units/self.units).to("dimensionless")
        if unit_conversion:
            other = float(unit_conversion)*other

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


def parse_subs(varkeys, substitutions, sweeps=True):
    "Seperates subs into constants, sweeps linkedsweeps actually present."
    varkeys.update_keymap()
    constants, sweep, linkedsweep = {}, None, None
    if sweeps:
        sweep, linkedsweep = {}, {}
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
    if sweeps:
        return constants, sweep, linkedsweep
    else:
        return constants


def append_sub(sub, keys, constants, sweep=None, linkedsweep=None):
    "Appends sub to constants, sweep, or linkedsweep."
    sweepsub = is_sweepvar(sub)
    if sweepsub and sweep is None and linkedsweep is None:
        return
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

        if hasattr(value, "__call__") and not hasattr(value, "key"):
            linkedsweep[key] = value
        elif sweepsub:
            sweep[key] = value
        else:
            try:
                assert np.isnan(value)
            except (AssertionError, TypeError, ValueError):
                constants[key] = value
