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
from .nomials import Posynomial, Monomial
from .varkey import VarKey

from .small_classes import Strings
from .small_scripts import flatten
from .small_scripts import locate_vars
from .small_scripts import mag
from .small_scripts import is_sweepvar

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
                setup = self.setup(*args, **kwargs)
            except:
                # TODO: pass error through!
                print("The 'setup' method of this model had an error.")
                raise TypeError
            try:
                cost, constraints = setup
            except TypeError:
                raise TypeError("Modek 'setup' methods must return "
                                "(cost, constraints).")
        self.cost = cost
        self.constraints = constraints
        self.substitutions = substitutions

        if hasattr(self, "setup"):
            # TODO: use super instead of Model
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
        for constraint in self.constraints:
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

    def solve(self, verbosity=2, skipfailures=True, *args, **kwargs):
        #if not all([isinstance(c, Posynomial) for c in constraints]):
        #    raise ValueError("'solve()' can only be called on models"
        #                     " that do not contain Signomials.")
        return self._solve("gp", verbosity, skipfailures, *args, **kwargs)

    def localsolve(self, verbosity=2, skipfailures=True, *args, **kwargs):
        #if all([isinstance(c, Posynomial) for c in constraints]):
        #    raise ValueError("'localsolve()' can only be called on models"
        #                     " that contain Signomials.")
        return self._solve("sp", verbosity, skipfailures, *args, **kwargs)

    def _solve(self, programType, verbosity, skipfailures, *args, **kwargs):
        posynomials = self.posynomials
        self.unsubbed_cs = np.hstack((mag(p.cs) for p in posynomials))
        self.unsubbed_exps = functools_reduce(add, (p.exps for p in posynomials))
        (self.unsubbed_varlocs,
         self.unsubbed_varkeys) = locate_vars(self.unsubbed_exps)

        (sweep, linkedsweep,
         constants) = separate_subs(self.substitutions, self.unsubbed_varkeys)
        solution = SolutionArray()
        kwargs.update({"printing": verbosity > 1})

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
                program, subs = self.formProgram(programType, constants_)

                try:
                    result = program.solve(*args, **kwargs)
                    sol = self.parse_result(result, subs,
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
            self.program, subs = self.formProgram(programType, constants)
            result = self.program.solve(*args, **kwargs)
            sol = self.parse_result(result, subs)
            solution.append(sol)

        solution.program = self.program
        solution.toarray()
        self.solution = solution
        return solution

    def sub(self, substitutions):
        substitutions = self.substitutions.update(substitutions)
        return Model(self.cost, self.constraints, substitutions)

    def formProgram(self, programType, constants):
        subs = {var: var.descr["value"]
                for var in self.variables if "value" in var.descr}
        subs.update(constants)
        cost = self.cost.sub(subs)
        constraints = [c.sub(subs) for c in self.constraints]

        if programType in ["gp", "GP"]:
            return GeometricProgram(cost, constraints), subs
        elif programType in ["sp", "SP"]:
            return SignomialProgram(cost, constraints), subs
        else:
            raise ValueError("unkonwn program type %s." % programType)

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

    def parse_result(self, result, constants, sweep={}, linkedsweep={}):
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

        nu = result["sensitivities"]["monomials"]
        sens_vars = {var: (sum([self.unsubbed_exps[i][var]*nu[i]
                                for i in locs]))
                     for (var, locs) in self.unsubbed_varlocs.items()}
        sens_vars.update({v: 0 for v in self.unsubbed_varlocs
                          if v not in sens_vars})
        sensitivities["variables"] = sens_vars

        # free-variable sensitivities must be < arbitrary epsilon 1e-4
        for var, S in sensitivities["variables"].items():
            if var in freevariables and abs(S) > 1e-4:
                print("free variable too sensitive:"
                                     " S_{%s} = %0.2e" % (var, S))

        localexp = {var: S for (var, S) in sens_vars.items() if abs(S) >= 0.1}
        print variables, localexp
        localcs = (variables[var]**-S for (var, S) in localexp.items())
        localc = functools_reduce(mul, localcs, cost)
        localmodel = Monomial(localexp, localc)

        # vectorvar substitution
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

                try:
                    variables[veckey][idx] = variables[var]
                    sensitivities["variables"][veckey][var.descr["idx"]] = \
                        sensitivities["variables"][var]
                except KeyError:
                    variables[veckey] = np.empty(var.descr["shape"]) + np.nan
                    sensitivities["variables"][veckey] = (
                        np.empty(var.descr["shape"]) + np.nan)

                    variables[veckey][idx] = variables[var]
                    sensitivities["variables"][veckey][var.descr["idx"]] = \
                        sensitivities["variables"][var]

                del variables[var]
                del sensitivities["variables"][var]

                for varlist in [constants, sweepvariables, freevariables]:
                    if var in varlist:
                        varlist[veckey] = variables[veckey]
                        del varlist[var]

        return dict(cost=cost,
                    constants=constants,
                    sweepvariables=sweepvariables,
                    freevariables=freevariables,
                    variables=variables,
                    sensitivities=sensitivities,
                    localmodel=localmodel)


def separate_subs(substitutions, varkeys):
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
                            linkedsweep.update({v: suba[i]})
                        else:
                            sweep.update({v: suba[i]})
                elif len(var) == suba.shape[1]:
                    raise ValueError("whole-vector substitution"
                                     "is not yet supported")
                else:
                    raise ValueError("vector substitutions must share a"
                                     "dimension with the variable vector")
            elif hasattr(sub[1], "__call__"):
                linkedsweep.update({var: sub[1]})
            else:
                sweep.update({var: sub[1]})
    return sweep, linkedsweep, constants
