import numpy as np

from .nomials import Monomial
from .geometric_program import GP
from .small_scripts import locate_vars
from .nomials import Constraint, MonoEQConstraint
from .small_classes import CootMatrix
from .small_scripts import mag
from collections import defaultdict


class SP(GP):

    def _run_solver(self):
        "Gets a solver's raw output, then checks and standardizes it."
        lastobj = 1
        obj = 1
        init = 2
        while abs(lastobj-obj)/(lastobj + obj) > 1e-4 or init:
            if init:
                init -= 1
            lastobj = obj
            cs, exps, A, p_idxs, k = self.genA()
            result = self.solverfn(c=cs, A=A, p_idxs=p_idxs, k=k)
            if result['status'] not in ["optimal", "OPTIMAL"]:
                raise RuntimeWarning("final status of solver '%s' was '%s' not "
                                     "'optimal'" % (self.solver, result['status']))
            self.xk = dict(zip(self.varlocs, np.exp(result['primal']).ravel()))
            if "objective" in result:
                obj = float(result["objective"])
            else:
                obj = self.cost.subcmag(self.xk)

        cs, p_idxs = map(np.array, [cs, p_idxs])
        return self._parse_result(result, senss=False, cs=cs, p_idxs=p_idxs)

    def genA(self, printing=True):
        # A: exponents of the various free variables for each monomial
        #    rows of A are variables, columns are monomials

        printing = printing and bool(self.xk)


        cs, exps, p_idxs = [], [], []
        varkeys = []
        approxs = {}
        for i in range(len(self.cs)):
            if self.cs[i] < 0:
                c = -self.cs[i]
                exp = self.exps[i]
                if self.xk:
                    for vk in exp:
                        if vk not in varkeys:
                            varkeys.append(vk)
                p_idx = self.p_idxs[i]
                m = Monomial(exp, c)
                if p_idx not in approxs:
                    approxs[p_idx] = m
                else:
                    approxs[p_idx] += m
            else:
                cs.append(self.cs[i])
                exps.append(self.exps[i])
                p_idxs.append(self.p_idxs[i])

        if self.xk:
            missing_vks = [vk for vk in varkeys if vk not in self.xk]
            if missing_vks:
                raise RuntimeWarning("starting point for solution needs to"
                                     "contain the following variables: "
                                     + str(missing_vks))

        for p_idx, p in approxs.items():
            approxs[p_idx] = (1+p).mono_approximation(self.xk)

        for i, p_idx in enumerate(p_idxs):
            if p_idx in approxs:
                cs[i] /= approxs[p_idx].c
                exps[i] -= approxs[p_idx].exp
                if mag(cs[i]) > 1 and not exps[i]:
                    # HACK: remove guaranteed-infeasible constraints
                    cs.pop(i)
                    exps.pop(i)
                    p_idxs.pop(i)

        k = []
        count = 1
        last = None
        for p in p_idxs:
            if last is not None:
                if last == p:
                    count += 1
                else:
                    k.append(count)
                    count = 1
            last = p
        k.append(count)

        varlocs, varkeys = locate_vars(exps)

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
            self.A.append(0, len(self.exps)-1, 0)
        self.A.update_shape()

        self.missingbounds = missingbounds
        if printing:
            self.checkbounds()

        return cs, exps, self.A, p_idxs, k

    def localsolve(self, printing=True, xk={}, *args, **kwargs):
        if printing:
            print "Beginning signomial local-solve:"

        self.xk = xk
        self.presolve = self.last

        return self._solve(printing=printing, *args, **kwargs)

    def solve(self, *args, **kwargs):
        raise KeyError("Signomial programs have only local solutions, and"
                       "must be solved using the the 'localsolve' function"
                       "instead of the 'solve' function.")
