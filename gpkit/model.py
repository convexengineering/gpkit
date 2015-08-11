# -*- coding: utf-8 -*-
"""Module for creating Model instances.

    Example
    -------
    >>> gp = gpkit.Model(cost, constraints, substitutions)

"""

import numpy as np

from time import time
from pprint import pformat
from collections import defaultdict
from functools import reduce as functools_reduce
from operator import mul, add
from copy import deepcopy
from collections import Iterable

from .nomials import Constraint, MonoEQConstraint
from .nomials import Posynomial, Monomial, Signomial
from .varkey import VarKey
from .substitution import substitution, getsubs

from .small_classes import Strings
from .small_scripts import flatten
from .small_scripts import locate_vars
from .small_scripts import mag
from .small_scripts import is_sweepvar
from .small_scripts import sort_and_simplify

from .solution_array import SolutionArray
from .signomial_program import SignomialProgram
from .geometric_program import GeometricProgram

from .variables import Variable

try:
    from IPython.parallel import Client
    CLIENT = Client(timeout=0.01)
    assert len(CLIENT) > 0
    POOL = CLIENT[:]
    POOL.use_dill()
    print("Using parallel execution of sweeps on %s clients" % len(CLIENT))
except:
    POOL = None


class Model(object):
    model_nums = defaultdict(int)

    def __init__(self, cost=None, constraints=[],
                 substitutions={}, *args, **kwargs):
        if cost is None:
            if not hasattr(self, "setup"):
                raise TypeError("Models can only be created without a cost"
                                " if they have a 'setup' method.")
            try:
                if "name" in kwargs:
                    name = kwargs.pop("name")
                else:
                    name = self.__class__.__name__
                setup = self.setup(*args, **kwargs)
            except:
                print("The 'setup' method of this model had an error.")
                raise
            try:
                cost, constraints = setup
            except TypeError:
                raise TypeError("Model 'setup' methods must return "
                                "(cost, constraints).")
        self.cost = cost
        self.constraints = list(constraints)
        self.substitutions = dict(substitutions)

        if hasattr(self, "setup"):
            # TODO: use super instead of Model?
            k = Model.model_nums[name]
            Model.model_nums[name] = k+1
            name += str(k) if k else ""
            for p in posynomials:
                for k in p.varlocs:
                    if "model" not in k.descr:
                        newk = VarKey(k, model=name)
                        p.varlocs[newk] = p.varlocs.pop(k)
                for exp in p.exps:
                    for k in exp:
                        if "model" not in k.descr:
                            newk = VarKey(k, model=name)
                            exp[newk] = exp.pop(k)

    @property
    def variables(self):
        variables = {}
        for p in self.posynomials:
            variables.update({vk: Variable(**vk.descr) for vk in p.varlocs})
        return variables

    @property
    def posynomials(self):
        constraints = tuple(flatten(self.constraints, Constraint))
        # TODO: parse constraints during flattening, calling Posyarray on
        #       anything that holds only posys and then saving that list.
        #       This will allow prettier constraint printing.
        posynomials = [self.cost]
        for constraint in constraints:
            if isinstance(constraint, MonoEQConstraint):
                posynomials += [constraint.leq, constraint.geq]
            else:
                posynomials.append(constraint)
        return posynomials

    def __add__(self, other):
        if isinstance(other, Model):
            substitutions = dict(self.substitutions)
            substitutions.update(other.substitutions)
            return Model(self.cost*other.cost,
                         self.constraints + other.constraints,
                         substitutions)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Model):
            # TODO: combine shared variables
            substitutions = dict(self.substitutions)
            substitutions.update(other.substitutions)
            return Model(self.cost*other.cost,
                         self.constraints + other.constraints,
                         substitutions)
        else:
            return NotImplemented

    # TODO: add get_item

    def solve(self, solver=None, verbosity=2, skipfailures=True, *args, **kwargs):
        #if not all([isinstance(c, Posynomial) for c in constraints]):
        #    raise ValueError("'solve()' can only be called on models"
        #                     " that do not contain Signomials.")
        return self._solve("gp", solver, verbosity, skipfailures, *args, **kwargs)

    def localsolve(self, solver=None, verbosity=2, skipfailures=True, *args, **kwargs):
        #if all([isinstance(c, Posynomial) for c in constraints]):
        #    raise ValueError("'localsolve()' can only be called on models"
        #                     " that contain Signomials.")
        return self._solve("sp", solver, verbosity, skipfailures, *args, **kwargs)

    def _solve(self, programType, solver, verbosity, skipfailures, *args, **kwargs):
        posynomials = self.posynomials
        self.unsubbed_cs = np.hstack((mag(p.cs) for p in posynomials))
        self.unsubbed_exps = functools_reduce(add, (p.exps for p in posynomials))
        (self.unsubbed_varlocs,
         self.unsubbed_varkeys) = locate_vars(self.unsubbed_exps)

        subs = {var: var.descr["value"]
                for var in self.variables if "value" in var.descr}
        subs.update(self.substitutions)

        (sweep, linkedsweep,
         constants) = separate_subs(subs, self.unsubbed_varkeys,
                                    self.unsubbed_varlocs)
        solution = SolutionArray()
        kwargs.update({"solver": solver})
        kwargs.update({"verbosity": verbosity - 1})

        if sweep:
            if len(sweep) == 1:
                sweep_grids = np.array(list(sweep.values()))
            else:
                sweep_grids = np.meshgrid(*list(sweep.values()))

            N_passes = sweep_grids[0].size
            sweep_vects = {var: grid.reshape(N_passes)
                           for (var, grid) in zip(sweep, sweep_grids)}

            if verbosity > 0:
                print("Solving for %i variables over %i passes." % (
                      len(self.variables), N_passes))

            def solve_pass(i):
                this_pass = {var: sweep_vect[i]
                             for (var, sweep_vect) in sweep_vects.items()}
                linked = {var: fn(*[this_pass[VarKey(v)]
                                    for v in var.descr["args"]])
                          for var, fn in linkedsweep.items()}
                this_pass.update(linked)
                constants_ = constants
                constants_.update(this_pass)
                program, subs, mmaps = self.formProgram(programType, posynomials,
                                                        constants_)

                try:
                    result = program.solve(*args, **kwargs)
                    sol = self.parse_result(result, subs, mmaps,
                                            sweep, linkedsweep)
                    return program, sol
                except (RuntimeWarning, ValueError):
                    return program, program.result

            if POOL:
                mapfn = POOL.map_sync
            else:
                mapfn = map

            self.program = []
            for program, result in mapfn(solve_pass, range(N_passes)):
                if not skipfailures:
                    self.program.append(program)
                    solution.append(result)
                elif not hasattr(result, "status"):
                    # this is an optimal solution
                    self.program.append(program)
                    solution.append(result)
        else:
            self.program, subs, mmaps = self.formProgram(programType,
                                                         posynomials,
                                                         constants)
            result = self.program.solve(*args, **kwargs)
            sol = self.parse_result(result, subs, mmaps)
            solution.append(sol)

        solution.program = self.program
        solution.toarray()
        self.solution = solution
        return solution

    def sub(self, substitutions, val=None):
        if val is not None:
            substitutions = {substitutions: val}
        self.substitutions.update(substitutions)
        # edits in place...should it return a new model instead?

    def formProgram(self, programType, signomials, subs):
        signomials_, mmaps = [], []
        for s in signomials:
            _, exps, cs, _ = substitution(s.varlocs, s.varkeys,
                                          s.exps, s.cs, subs)
            if any((mag(c) != 0 for c in cs)):
                exps, cs, mmap = sort_and_simplify(exps, cs, return_map=True)
                signomials_.append(Signomial(exps, cs, units=s.units))
                mmaps.append(mmap)
            else:
                mmaps.append([None]*len(cs))

        cost = signomials_[0]
        constraints = signomials_[1:]

        if programType in ["gp", "GP"]:
            return GeometricProgram(cost, constraints), subs, mmaps
        elif programType in ["sp", "SP"]:
            return SignomialProgram(cost, constraints), subs, mmaps
        else:
            raise ValueError("unknown program type %s." % programType)

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))

    def __str__(self):
        """String representation of a Model.
        Contains all of its parameters."""
        subs = {var: var.descr["value"]
                for var in self.variables if "value" in var.descr}
        subs.update(self.substitutions)
        return "\n".join(["# minimize",
                          "    %s," % self.cost,
                          "[   # subject to"] +
                         ["    %s," % constr
                          for constr in self.constraints] +
                         ['],',
                          "    substitutions={ %s }" %
                          pformat(subs, indent=20)[20:-1]])

    def _latex(self, unused=None):
        """LaTeX representation of a GeometricProgram.
        Contains all of its parameters."""
        sweep, linkedsweep,
        constants = separate_subs(self.substitutions, self.varkeys)
        subs = {var: var.descr["value"]
                for var in self.variables if "value" in var.descr}
        subs.update(constants)
        # TODO: print sweeps and linkedsweeps
        return "\n".join(["\\begin{array}[ll]",
                          "\\text{}",
                          "\\text{minimize}",
                          "    & %s \\\\" % self.cost._latex(),
                          "\\text{subject to}"] +
                         ["    & %s \\\\" % constr._latex()
                          for constr in self.constraints] +
                         ["\\text{substituting}"] +
                         sorted(["    & %s = %s \\\\" % (var._latex(),
                                                         latex_num(val))
                                 for var, val in subs.items()]) +
                         ["\\end{array}"])

    def parse_result(self, result, constants, mmaps, sweep={}, linkedsweep={}):
        cost = result["cost"]
        freevariables = dict(result["variables"])
        sweepvariables = {var: val for var, val in constants.items()
                          if var in sweep or var in linkedsweep}
        constants = {var: val for var, val in constants.items()
                     if var not in sweepvariables}
        variables = dict(freevariables)
        variables.update(constants)
        variables.update(sweepvariables)
        sensitivities = dict(result["sensitivities"])

        # Remap monomials after substitution and simplification.
        #  The monomial sensitivities from the GP/SP are in terms of this
        #  smaller post-substitution list of monomials, so we need to map that
        #  back to the pre-substitution list.
        #
        #  Each "mmap" is a list whose elements are either None
        #  (indicating that the monomial was removed after subsitution)
        #  or a tuple with:
        #    the index of the monomial they became after subsitution,
        #    the percentage of that monomial that they formed
        #      (equal to their c/that monomial's final c)
        nu = result["sensitivities"]["monomials"]
        nu_ = []
        mons = 0
        for mmap in mmaps:
            max_idx = 0
            for m_i in mmap:
                if m_i is None:
                    nu_.append(0)
                else:
                    idx, percentage = m_i
                    nu_.append(percentage*nu[idx + mons])
                    if idx > max_idx:
                        max_idx = idx
            mons += 1 + max_idx
        nu_ = np.array(nu_)
        sensitivities["monomials"] = nu_

        sens_vars = {var: sum([self.unsubbed_exps[i][var]*nu_[i]
                               for i in locs])
                     for (var, locs) in self.unsubbed_varlocs.items()}
        sensitivities["variables"] = sens_vars

        # free-variable sensitivities must be < arbitrary epsilon 1e-4
        for var, S in sensitivities["variables"].items():
            if var in freevariables and abs(S) > 1e-4:
                print("free variable too sensitive:"
                      " S_{%s} = %0.2e" % (var, S))

        localexp = {var: S for (var, S) in sens_vars.items() if abs(S) >= 0.1}
        localcs = (variables[var]**-S for (var, S) in localexp.items())
        localc = functools_reduce(mul, localcs, cost)
        localmodel = Monomial(localexp, localc)

        # vectorvar substitution
        veckeys = set()
        for var in self.unsubbed_varlocs:
            if "idx" in var.descr and "shape" in var.descr:
                descr = dict(var.descr)
                idx = descr.pop("idx")
                if "value" in descr:
                    descr.pop("value")
                if "units" in descr:
                    units = descr.pop("units")
                    veckey = VarKey(**descr)
                    veckey.descr["units"] = units
                else:
                    veckey = VarKey(**descr)
                veckeys.add(veckey)

                for vardict in [variables, sensitivities["variables"],
                                constants, sweepvariables, freevariables]:
                    if var in vardict:
                        if veckey in vardict:
                            vardict[veckey][idx] = vardict[var]
                        else:
                            vardict[veckey] = np.full(var.descr["shape"], np.nan)
                            vardict[veckey][idx] = vardict[var]

                        del vardict[var]

        # for veckey in veckeys:
        #     # TODO: print index that error occured at
        #     if any(np.isnan(variables[veckey])):
        #         print variables
        #         raise RuntimeWarning("did not fully fill vector variables.")

        return dict(cost=cost,
                    constants=constants,
                    sweepvariables=sweepvariables,
                    freevariables=freevariables,
                    variables=variables,
                    sensitivities=sensitivities,
                    localmodel=localmodel)


def separate_subs(substitutions, varkeys, varlocs):
    # TODO: refactor this
    sweep, linkedsweep = {}, {}
    constants = dict(substitutions)
    for var, sub in substitutions.items():
        if is_sweepvar(sub):
            del constants[var]
            if isinstance(var, Strings):
                var = varkeys[var]
            elif isinstance(var, Monomial):
                var = VarKey(var)
            if isinstance(var, Iterable):
                suba = np.array(sub[1])
                if len(var) == suba.shape[0]:
                    for i, v in enumerate(var):
                        if hasattr(suba[i], "__call__"):
                            linkedsweep.update({VarKey(v): suba[i]})
                        else:
                            sweep.update({VarKey(v): suba[i]})
                elif len(var) == suba.shape[1]:
                    raise ValueError("whole-vector substitution"
                                     " is not yet supported")
                else:
                    raise ValueError("vector substitutions must share a"
                                     "dimension with the variable vector")
            elif hasattr(sub[1], "__call__"):
                linkedsweep.update({var: sub[1]})
            else:
                sweep.update({var: sub[1]})
    constants = getsubs(varkeys, varlocs, constants)
    return sweep, linkedsweep, constants
