# -*- coding: utf-8 -*-
"Module containing the substitution function"

from collections import defaultdict, Iterable
import numpy as np
from .nomial_math import Monomial
from ..small_classes import Numbers, Strings, Quantity
from ..small_classes import HashVector
from ..varkey import VarKey
from ..small_scripts import is_sweepvar
from ..small_scripts import mag
from .. import DimensionalityError


def parse_subs(varkeys, substitutions):
    constants, sweep, linkedsweep = {}, {}, {}
    if hasattr(substitutions, "keymap"):
        keys_added = set()
        for var in varkeys.keymap:
            if var in substitutions.keymap:
                keys = substitutions.keymap[var]
                if len(keys) == 1:
                    key, = keys
                    if key not in keys_added:
                        keys_added.add(key)
                        sub = dict.__getitem__(substitutions, key)
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
            except:
                constants[key] = value
        elif not hasattr(value, "__call__"):
            sweep[key] = value
        else:
            linkedsweep[key] = value


def substitution(nomial, substitutions, val=None):
    """Efficient substituton into a list of monomials.

        Arguments
        ---------
        varlocs : dict
            Dictionary mapping variables to lists of monomial indices.
        exps : Iterable of dicts
            Dictionary mapping variables to exponents, for each monomial.
        cs : list
            Coefficient for each monomial.
        substitutions : dict
            Substitutions to apply to the above.
        val : number (optional)
            Used to substitute singlet variables.

        Returns
        -------
        varlocs_ : dict
            Dictionary of monomial indexes for each variable.
        exps_ : dict
            Dictionary of variable exponents for each monomial.
        cs_ : list
            Coefficients each monomial.
        subs_ : dict
            Substitutions to apply to the above.
    """

    if val is not None:
        substitutions = {substitutions: val}

    if not substitutions:
        return nomial.varlocs, nomial.exps, nomial.cs, substitutions

    subs, _, _ = parse_subs(nomial.varkeys, substitutions)

    if not subs:
        return nomial.varlocs, nomial.exps, nomial.cs, subs

    exps_ = [HashVector(exp) for exp in nomial.exps]
    cs_ = np.array(nomial.cs)
    if nomial.units:
        cs_ = Quantity(cs_, nomial.cs.units)
    varlocs_ = defaultdict(list)
    varlocs_.update({vk: list(idxs) for (vk, idxs) in nomial.varlocs.items()})
    for var, sub in subs.items():
        for i in nomial.varlocs[var]:
            x = exps_[i].pop(var)
            varlocs_[var].remove(i)
            if len(varlocs_[var]) == 0:
                del varlocs_[var]
            if isinstance(sub, Numbers):
                if sub != 0:
                    if nomial.units:
                        cs_.magnitude[i] *= sub**x
                    else:
                        cs_[i] *= sub**x
                else:  # frickin' pints bug. let's reimplement pow()
                    if x > 0:
                        mag(cs_)[i] = 0.0
                    elif x < 0:
                        if mag(cs_[i]) > 0:
                            mag(cs_)[i] = np.inf
                        elif mag(cs_[i]) < 0:
                            mag(cs_)[i] = -np.inf
                        else:
                            mag(cs_)[i] = np.nan
                    else:
                        mag(cs_)[i] = 1.0
            elif isinstance(sub, np.ndarray):
                if not sub.shape:
                    cs_[i] *= sub.flatten()[0]**x
            elif isinstance(sub, Strings):
                descr = dict(var.descr)
                del descr["name"]
                sub = VarKey(name=sub, **descr)
                exps_[i] += HashVector({sub: x})
                varlocs_[sub].append(i)
            elif isinstance(sub, (VarKey, Monomial)):
                if isinstance(var.units, Quantity):
                    if sub.units != var.units:
                        try:
                            cs_[i] *= (sub.units/var.units).to('dimensionless')
                        except DimensionalityError:
                            raise ValueError("units of the substituted %s '%s'"
                                             " [%s] are not compatible with"
                                             " those of the original '%s'"
                                             " [%s]." %
                                             (type(sub),
                                              sub.str_without(["units"]),
                                              sub.units.units,
                                              var, var.units.units))
                if isinstance(sub, VarKey):
                    exps_[i] += HashVector({sub: x})
                    varlocs_[sub].append(i)
                else:
                    exps_[i] += x*sub.exp
                    cs_[i] *= mag(sub.c)**x
                    for subvar in sub.exp:
                        varlocs_[subvar].append(i)
            else:
                raise TypeError("could not substitute with value"
                                " of type '%s'" % type(sub))
    return varlocs_, exps_, cs_, subs
