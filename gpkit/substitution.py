# -*- coding: utf-8 -*-
"Module containing the substitution function"

import numpy as np

from collections import defaultdict
from collections import Iterable

from .small_classes import Numbers, Strings
from .small_classes import HashVector
from .nomials import Monomial
from .nomials import VarKey
from .nomials import VectorVariable

from .small_scripts import invalid_types_for_oper
from .small_scripts import locate_vars
from .small_scripts import is_sweepvar


def vectorsub(subs, var, sub, varset):
    "Vectorized substitution via vecmons and Variables."
    try:
        isvector = "length" in var.descr and "idx" not in var.descr
    except:
        try:
            assert len(var)
            isvector = True
        except:
            isvector = False

    if var in varset:
        subs[var] = sub
    elif isvector:
        if isinstance(var, VarKey):
            var = VectorVariable(**var.descr)
        if len(var) == len(sub):
            for i in range(len(var)):
                v = VarKey(var[i])
                if v in varset:
                    subs[v] = sub[i]
        else:
            raise ValueError("tried substituting %s for %s, but their"
                             " lengths were unequal." % (sub, var))


def substitution(var_locs, exps, cs, substitutions, val=None):
    """Efficient substituton into a list of monomials.

        Parameters
        ----------
        var_locs : dict
            Dictionary of monomial indexes for each variable.
        exps : dict
            Dictionary of variable exponents for each monomial.
        cs : list
            Coefficients each monomial.
        substitutions : dict
            Substitutions to apply to the above.
        val : number (optional)
            Used to substitute singlet variables.

        Returns
        -------
        var_locs_ : dict
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

    subs = {}
    varset = frozenset(var_locs.keys())
    for var, sub in substitutions.items():
        if not is_sweepvar(sub):
            if isinstance(var, Strings+(Monomial,)):
                var_ = VarKey(var)
                if var_ in varset:
                    subs[var_] = sub
            else:
                vectorsub(subs, var, sub, varset)

    if not subs:
        raise KeyError("could not find anything to substitute.")

    exps_ = [HashVector(exp) for exp in exps]
    cs_ = np.array(cs)
    var_locs_ = defaultdict(list)
    var_locs_.update({var: list(idxs) for (var, idxs) in var_locs.items()})
    for var, sub in subs.items():
        for i in var_locs[var]:
            x = exps_[i].pop(var)
            var_locs_[var].remove(i)
            if len(var_locs_[var]) == 0:
                del var_locs_[var]
            if isinstance(sub, Numbers):
                cs_[i] *= sub**x
            elif isinstance(sub, np.ndarray):
                if not sub.shape:
                    cs_[i] *= sub.flatten()[0]**x
            #  BELOW DOES NOT SUPPORT UNIT CONVERSION YET
            # elif isinstance(sub, Strings+(VarKey,)):
            #     sub = VarKey(sub)
            #     exps_[i] += HashVector({sub: x})
            #     var_locs_[sub].append(i)
            # elif isinstance(sub, Monomial):
            #     exps_[i] += x*sub.exp
            #     cs_[i] *= sub.c**x
            #     for subvar in sub.exp:
            #         var_locs_[subvar].append(i)
    return var_locs_, exps_, cs_, subs
