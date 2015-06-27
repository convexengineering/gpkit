# -*- coding: utf-8 -*-
"""Module for creating models.

    Currently these are only used for GP instances, but they may be further
    abstractable.

"""

import numpy as np

from functools import reduce as functools_reduce
from operator import add
from collections import Iterable

from .small_classes import Strings
from .small_classes import PosyTuple
from .small_classes import CootMatrix
from .nomials import Constraint, Monomial
from .varkey import VarKey

from .substitution import substitution
from .small_scripts import locate_vars
from .small_scripts import is_sweepvar
from .small_scripts import mag


class Model(object):
    "Abstract class with substituion, loading / saving, and p_idx/A generation"

    def __repr__(self):
        return "\n".join(["gpkit.Model with",
                          "  Equations"] +
                         ["     %s <= 1" % p._string()
                          for p in self.posynomials])

    def add_constraints(self, constraints):
        if isinstance(constraints, Constraint):
            constraints = [constraints]
        self.constraints += tuple(constraints)
        self._gen_unsubbed_vars()

    def rm_constraints(self, constraints):
        if isinstance(constraints, Constraint):
            constraints = [constraints]
        for p in constraints:
            self.constraints.remove(p)
        self._gen_unsubbed_vars()

    def _gen_unsubbed_vars(self, exps=None, cs=None):
        posynomials = self.posynomials
        if not exps and not cs:
            exps = functools_reduce(add, (x.exps for x in posynomials))
            cs = np.hstack((mag(p.cs) for p in posynomials))
        varlocs, varkeys = locate_vars(exps)

        self.unsubbed = PosyTuple(exps, cs, varlocs, {})
        self.variables = {k: Monomial(k) for k in varlocs}
        self.varkeys = varkeys
        self.load(self.unsubbed)

        # k [j]: number of monomials (columns of F) present in each constraint
        self.k = [len(p.cs) for p in posynomials]

        # p_idxs [i]: posynomial index of each monomial
        p_idx = 0
        self.p_idxs = []
        for p_len in self.k:
            self.p_idxs += [p_idx]*p_len
            p_idx += 1
        self.p_idxs = np.array(self.p_idxs)

    def sub(self, substitutions, val=None, frombase='last', replace=True):
        """Substitutes into a model, modifying it in place.

        Usage
        -----
        gp.sub({'x': 1, y: 2})
        gp.sub(x, 3, replace=True)

        Arguments
        ---------
        substitutions : dict or key
            Either a dictionary whose keys are strings, Variables, or VarKeys, 
            and whose values are numbers, or a string, Variable or Varkey.
        val : number (optional)
            If the substitutions entry is a single key, val holds the value
        frombase : string (optional)
            Which model state to update. By default updates the current state.
        replace : boolean (optional, default is True)
            Whether the substitution should only subtitute currently
            unsubstituted values (False) or should also make replacements of
            current substitutions (True).

        Returns
        -------
        No return value: the model is modified in place.
        """
        # look for sweep variables
        found_sweep = []
        if val is not None:
            substitutions = {substitutions: val}
        subs = dict(substitutions)
        for var, sub in substitutions.items():
            if is_sweepvar(sub):
                del subs[var]
                if isinstance(var, Strings):
                    var = self.varkeys[var]
                elif isinstance(var, Monomial):
                    var = VarKey(var)
                if isinstance(var, Iterable):
                    suba = np.array(sub[1])
                    if len(var) == suba.shape[0]:
                        for i, v in enumerate(var):
                            found_sweep.append(v)
                            if hasattr(suba[i], "__call__"):
                                self.linkedsweep.update({v: suba[i]})
                            else:
                                self.sweep.update({v: suba[i]})
                    elif len(var) == suba.shape[1]:
                        raise ValueError("whole-vector substitution"
                                         "is not yet supported")
                    else:
                        raise ValueError("vector substitutions must share a"
                                         "dimension with the variable vector")
                elif hasattr(sub[1], "__call__"):
                    found_sweep.append(var)
                    self.linkedsweep.update({var: sub[1]})
                else:
                    found_sweep.append(var)
                    self.sweep.update({var: sub[1]})

        if not (subs or found_sweep):
            raise KeyError("could not find anything"
                           "to substitute in %s" % substitutions)

        base = getattr(self, frombase)
        substitutions = dict(base.substitutions)

        if replace:
            (varlocs,
             exps,
             cs) = self.unsubbed.varlocs, self.unsubbed.exps, self.unsubbed.cs
            for sweepvar in found_sweep:
                if sweepvar in substitutions:
                    del substitutions[sweepvar]
            if subs:
                # resubstitute from the beginning, in reverse order
                varlocs, exps, cs, subs = substitution(varlocs,
                                                       self.varkeys,
                                                       exps,
                                                       cs,
                                                       subs)
                substitutions.update(subs)
            try:
                varlocs, exps, cs, subs = substitution(varlocs,
                                                       self.varkeys,
                                                       exps,
                                                       cs,
                                                       substitutions)
            except KeyError:
                # our new sub replaced the only previous sub
                pass
        else:
            varlocs, exps, cs = base.varlocs, base.exps, base.cs
            # substitute normally
            if subs:
                varlocs, exps, cs, subs = substitution(varlocs,
                                                       exps,
                                                       cs,
                                                       subs)
            substitutions.update(subs)

        self.load(PosyTuple(exps, cs, varlocs, substitutions))

    def load(self, posytuple):
        self.last = posytuple
        for attr in ['exps', 'cs', 'varlocs', 'substitutions']:
            setattr(self, attr, getattr(posytuple, attr))

    def genA(self, allpos=True, printing=True):
        # A: exponents of the various free variables for each monomial
        #    rows of A are variables, columns are monomials

        removed_idxs = []

        if not allpos:
            cs, exps, p_idxs = [], [], []
            for i in range(len(self.cs)):
                if self.cs[i] < 0:
                    raise RuntimeWarning("GPs cannot have negative "
                                         "coefficients")
                elif self.cs[i] > 0:
                    cs.append(self.cs[i])
                    exps.append(self.exps[i])
                    p_idxs.append(self.p_idxs[i])
                else:
                    removed_idxs.append(i)

            k = []
            count = 1
            last = None
            for p in p_idxs:
                if last == p:
                    count += 1
                elif last is not None:
                    k.append(count)
                    count = 1
                last = p
            k.append(count)

            varlocs, _ = locate_vars(exps)
        else:
            cs, exps, p_idxs, k, varlocs = (self.cs, self.exps, self.p_idxs,
                                            self.k, self.varlocs)

        missingbounds = {}
        self.A = CootMatrix([], [], [])
        for j, var in enumerate(varlocs):
            varsign = "both" if "value" in var.descr else None
            for i in varlocs[var]:
                exp = exps[i][var]
                self.A.append(i, j, exp)
                if varsign is "both":
                    pass
                elif varsign is None:
                    varsign = np.sign(exp)
                elif np.sign(exp) != varsign:
                    varsign = "both"

            if varsign != "both" and var not in self.sweep:
                if varsign == 1:
                    bound = "lower"
                elif varsign == -1:
                    bound = "upper"
                missingbounds[var] = bound

        # add subbed-out monomials at the end
        if not exps[-1]:
            self.A.append(0, len(exps)-1, 0)
        self.A.update_shape()

        self.missingbounds = missingbounds
        if printing:
            self.checkbounds()

        return cs, exps, self.A, p_idxs, k, removed_idxs

    def checkbounds(self):
        for var, bound in sorted(self.missingbounds.items()):
            print("%s has no %s bound" % (var, bound))
