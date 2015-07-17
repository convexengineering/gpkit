# -*- coding: utf-8 -*-
"Module containing the substitution function"

import numpy as np

from collections import defaultdict

from .small_classes import Numbers, Strings, Quantity
from .small_classes import HashVector
from .nomials import Monomial
from .varkey import VarKey
from .variables import VectorVariable

from .small_scripts import is_sweepvar
from .small_scripts import mag

from . import DimensionalityError


def getsubs(varkeys, varlocs, substitutions):
    subs = {}
    varset = frozenset(varlocs.keys())
    for var, sub in substitutions.items():
        if not is_sweepvar(sub):
            if isinstance(var, Monomial):
                var_ = VarKey(var)
                if var_ in varset:
                    subs[var_] = sub
            elif isinstance(var, Strings):
                if var in varkeys:
                    var_ = varkeys[var]
                    vectorsub(subs, var_, sub, varset)
            else:
                vectorsub(subs, var, sub, varset)
    return subs


def vectorsub(subs, var, sub, varset):
    "Vectorized substitution"

    if hasattr(var, "__len__"):
        isvector = True
    elif hasattr(var, "descr"):
        isvector = "shape" in var.descr
    else:
        isvector = False

    if isvector:
        if isinstance(sub, VarKey):
            sub = VectorVariable(**sub.descr)
        if hasattr(sub, "__len__"):
            if hasattr(sub, "shape"):
                isvector = bool(sub.shape)
        else:
            isvector = False

    if isvector:
        sub = np.array(sub)
        var = np.array(var)
        it = np.nditer(var, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            i = it.multi_index
            it.iternext()
            var_ = var[i]
            if var_ is not 0:
                v = VarKey(var_)
                if v in varset:
                    subs[v] = sub[i]
    elif var in varset:
        subs[var] = sub


def substitution(varlocs, varkeys, exps, cs, substitutions, val=None):
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

    subs = getsubs(varkeys, varlocs, substitutions)

    if not subs:
        return varlocs, exps, cs, subs
        # raise KeyError("could not find anything to substitute"
        #                "in %s" % substitutions)

    exps_ = [HashVector(exp) for exp in exps]
    cs_ = np.array(cs)
    varlocs_ = defaultdict(list)
    varlocs_.update({var: list(idxs) for (var, idxs) in varlocs.items()})
    for var, sub in subs.items():
        for i in varlocs[var]:
            x = exps_[i].pop(var)
            varlocs_[var].remove(i)
            if len(varlocs_[var]) == 0:
                del varlocs_[var]
            if isinstance(sub, Numbers):
                cs_[i] *= sub**x
            elif isinstance(sub, np.ndarray):
                if not sub.shape:
                    cs_[i] *= sub.flatten()[0]**x
            elif isinstance(sub, Strings):
                descr = dict(var.descr)
                del descr["name"]
                sub = VarKey(name=sub, **descr)
                exps_[i] += HashVector({sub: x})
                varlocs_[sub].append(i)
            elif isinstance(sub, VarKey):
                sub = VarKey(sub)
                if isinstance(var.units, Quantity):
                    try:
                        new_units = var.units/sub.units
                        cs_[i] *= new_units.to('dimensionless')
                    except DimensionalityError:
                        raise ValueError("substituted variables need the same"
                                         " units as variables they replace.")
                exps_[i] += HashVector({sub: x})
                varlocs_[sub].append(i)
            elif isinstance(sub, Monomial):
                if isinstance(var.units, Quantity):
                    try:
                        new_units = var.units/sub.units
                        cs_[i] *= new_units.to('dimensionless')
                    except DimensionalityError:
                        raise ValueError("substituted monomials need the same"
                                         " units as monomials they replace.")
                exps_[i] += x*sub.exp
                cs_[i] *= mag(sub.c)**x
                for subvar in sub.exp:
                    varlocs_[subvar].append(i)
            else:
                raise TypeError("could not substitute with value"
                                " of type '%s'" % type(sub))
    return varlocs_, exps_, cs_, subs
