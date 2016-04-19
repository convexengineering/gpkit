# -*- coding: utf-8 -*-
"Module containing the substitution function"

from collections import defaultdict, Iterable
import numpy as np
from ..small_classes import Numbers, Strings, Quantity
from ..small_classes import HashVector
from ..varkey import VarKey
from ..small_scripts import is_sweepvar
from ..small_scripts import mag
from .. import DimensionalityError


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
                if hasattr(sub, "units") and hasattr(sub, "to"):
                    if sub.units != var.units:
                        try:
                            vu = getattr(var.units, "units", "dimensionless")
                            sub = sub.to(vu)
                        except DimensionalityError:
                            raise ValueError("the units of '%s' are"
                                             " not compatible with those of"
                                             " those of the original '%s'"
                                             " [%s]." % (sub, var, vu))
                    sub = sub.magnitude
                # NOTE: uncomment the below to require Quantity'd subs
                # elif hasattr(var.units, "units"):
                #     try:
                #         sub /= var.units.to("dimensionless").magnitude
                #     except DimensionalityError:
                #         raise ValueError("cannot substitute the unitless"
                #                          " '%s' into '%s' of units '%s'." %
                #                          (sub, var, var.units.units))
                if sub != 0:
                    mag(cs_)[i] *= sub**x
                elif x > 0:  # HACK to prevent RuntimeWarnings
                    mag(cs_)[i] = 0
                elif x < 0:
                    if mag(cs_[i]) > 0:
                        mag(cs_)[i] = np.inf
                    elif mag(cs_[i]) < 0:
                        mag(cs_)[i] = -np.inf
                    else:
                        mag(cs_)[i] = np.nan
                # if sub is 0 and x is 0, pass
            elif isinstance(sub, np.ndarray):
                if not sub.shape:
                    cs_[i] *= sub.flatten()[0]**x
            elif isinstance(sub, Strings):
                descr = dict(var.descr)
                del descr["name"]
                sub = VarKey(name=sub, **descr)
                exps_[i] += HashVector({sub: x})
                varlocs_[sub].append(i)
            elif (isinstance(sub, VarKey)
                  or (hasattr(sub, "exp") and hasattr(sub, "c"))):
                if sub.units != var.units:
                    try:
                        if hasattr(sub.units, "to"):
                            vu = getattr(var.units, "units", "dimensionless")
                            mag(cs_)[i] *= mag(sub.units.to(vu))
                        elif hasattr(var.units, "to"):
                            units = sub.units if sub.units else "dimensionless"
                            mag(cs_)[i] /= mag(var.units.to(units))
                    except DimensionalityError:
                        raise ValueError("units of the substituted %s '%s'"
                                         " [%s] are not compatible with"
                                         " those of the original '%s' [%s]." %
                                         (type(sub),
                                          sub.str_without(["units"]),
                                          sub.units.units,
                                          var, var.units.units))
                if isinstance(sub, VarKey):
                    exps_[i][sub] = x + exps_[i].get(x, 0)
                    varlocs_[sub].append(i)
                else:
                    exps_[i] += x*sub.exp
                    mag(cs_)[i] *= mag(sub.c)**x
                    for subvar in sub.exp:
                        varlocs_[subvar].append(i)
            else:
                raise TypeError("could not substitute with value"
                                " of type '%s'" % type(sub))
    return varlocs_, exps_, cs_, subs
