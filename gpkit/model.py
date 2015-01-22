# -*- coding: utf-8 -*-
"""Module for creating models.

    Currently these are only used for GP instances, but they may be further
    abstractable.

"""

import numpy as np

from functools import reduce
from operator import add

from .small_classes import Strings
from .small_classes import PosyTuple
from .small_classes import CootMatrix
from .nomials import Posynomial
from .nomials import VarKey

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
                          for p in self.posynomials]
                         )

    def add_constraints(self, constraints):
        if isinstance(constraints, Posynomial):
            constraints = [constraints]
        self.constraints += tuple(constraints)
        self._gen_unsubbed_vars()

    def rm_constraints(self, constraints):
        if isinstance(constraints, Posynomial):
            constraints = [constraints]
        for p in constraints:
            self.constraints.remove(p)
        self._gen_unsubbed_vars()

    def _gen_unsubbed_vars(self, printing=False):
        posynomials = self.posynomials

        exps = reduce(add, map(lambda x: x.exps, posynomials))
        cs = np.hstack(map(lambda p: mag(p.cs), posynomials))
        var_locs = locate_vars(exps)

        self.unsubbed = PosyTuple(exps, cs, var_locs, {})
        self.load(self.unsubbed, printing)

        # k [j]: number of monomials (columns of F) present in each constraint
        self.k = [len(p.cs) for p in posynomials]

        # p_idxs [i]: posynomial index of each monomial
        p_idx = 0
        self.p_idxs = []
        for p_len in self.k:
            self.p_idxs += [p_idx]*p_len
            p_idx += 1
        self.p_idxs = np.array(self.p_idxs)

    def sub(self, substitutions, val=None, frombase='last', printing=False, replace=False):
        # look for sweep variables
        found_sweep = []
        if val is not None:
            substitutions = {substitutions: val}
        subs = dict(substitutions)
        for var, sub in substitutions.items():
            if is_sweepvar(sub):
                del subs[var]
                var = VarKey(var)
                found_sweep.append(var)
                self.sweep.update({var: sub[1]})

        if not (subs or found_sweep):
            raise KeyError("could not find anything to substitute")

        base = getattr(self, frombase)
        substitutions = dict(base.substitutions)

        if replace:
            var_locs, exps, cs = self.unsubbed.var_locs, self.unsubbed.exps, self.unsubbed.cs
            for sweepvar in found_sweep:
                if sweepvar in substitutions:
                    del substitutions[sweepvar]
            if subs:
                # resubstitute from the beginning, in reverse order
                var_locs, exps, cs, subs = substitution(var_locs,
                                                        exps,
                                                        cs,
                                                        subs)
                substitutions.update(subs)
            try:
                var_locs, exps, cs, subs = substitution(var_locs,
                                                        exps,
                                                        cs,
                                                        substitutions)
            except KeyError:
                # our new sub replaced the only previous sub
                pass
        else:
            var_locs, exps, cs = base.var_locs, base.exps, base.cs
            # substitute normally
            if subs:
                var_locs, exps, cs, subs = substitution(var_locs,
                                                        exps,
                                                        cs,
                                                        subs)
            substitutions.update(subs)

        self.load(PosyTuple(exps, cs, var_locs, substitutions), printing)

    def load(self, posytuple, printing=True):
        self.last = posytuple
        for attr in ['exps', 'cs', 'var_locs', 'substitutions']:
            setattr(self, attr, getattr(posytuple, attr))

        # A: exponents of the various free variables for each monomial
        #    rows of A are variables, columns are monomials
        missingbounds = {}
        self.A = CootMatrix([], [], [])
        for j, var in enumerate(self.var_locs):
            varsign = "both" if "value" in var.descr else None
            for i in self.var_locs[var]:
                exp = self.exps[i][var]
                self.A.append(i, j, exp)
                if varsign is "both": pass
                elif varsign is None: varsign = np.sign(exp)
                elif np.sign(exp) != varsign: varsign = "both"

            if varsign != "both" and var not in self.sweep:
                if varsign == 1: bound = "lower"
                elif varsign == -1: bound = "upper"
                missingbounds[var] = bound

        # add subbed-out monomials at the end
        if not self.exps[-1]:
            self.A.append(0, len(self.exps)-1, 0)
        self.A.update_shape()

        if printing:
            for var, bound in missingbounds.items():
                print("%s has no %s bound" % (var, bound))
