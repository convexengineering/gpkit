# -*- coding: utf-8 -*-
"Module containing miscellanous useful functions"

from collections import defaultdict
from collections import Iterable

from .small_classes import Numbers, Strings
from .small_classes import HashVector
from .nomials import Monomial
from .nomials import Variable
from .nomials import monovector

from .small_scripts import invalid_types_for_oper
from .small_scripts import locate_vars
from .small_scripts import is_sweepvar


def vectorsub(subs, var, subiter, var_locs):
    "Vectorized substitution via monovectors and Variables."
    if isinstance(var, Variable):
        var = monovector(**var.descr)
    if len(var) == len(subiter):
        for i in range(len(var)):
            v = Variable(var[i])
            if v in var_locs:
                subs[v] = subiter[i]
    else:
        raise ValueError("tried substituting %s for %s, but their"
                         "lengths were unequal." % (sub, var))


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
    for var, sub in substitutions.items():
        if isinstance(sub, Iterable):
            if len(sub) == 1:
                sub = sub[0]
        if not is_sweepvar(sub):
            if isinstance(sub, Iterable):
                vectorsub(subs, var, sub, var_locs)
            elif var in var_locs:
                subs[var] = sub
            elif Variable(var) in var_locs:
                subs[Variable(var)] = sub

    if not subs:
        raise KeyError("could not find anything to substitute.")

    exps_ = [HashVector(exp) for exp in exps]
    cs_ = list(cs)
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
            elif isinstance(sub, Strings+(Variable,)):
                sub = Variable(sub)
                exps_[i] += HashVector({sub: x})
                var_locs_[sub].append(i)
            elif isinstance(sub, Monomial):
                exps_[i] += x*sub.exp
                cs_[i] *= sub.c**x
                for subvar in sub.exp:
                    var_locs_[subvar].append(i)
    return var_locs_, exps_, cs_, subs
