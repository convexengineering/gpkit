# -*- coding: utf-8 -*-
"""Module for creating Model instances.

    Example
    -------
    >>> gp = gpkit.Model(cost, constraints, substitutions)

"""

import numpy as np

from pprint import pformat
from collections import defaultdict
from functools import reduce as functools_reduce
from operator import mul, add
from collections import Iterable

from .nomials import Constraint, MonoEQConstraint, Posynomial
from .nomials import Monomial, Signomial
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
    """Symbolic representation of an optimization problem.

    The Model class is used both directly to create models with constants and
    sweeps, and indirectly inherited to create custom model classes.

    Arguments
    ---------
    cost : Signomial (optional)
        If this is undeclared, the Model will get its cost and constraints
        from its "setup" method. This allows for easy inheritance.

    constraints : list of Constraints (optional)
        Defaults to an empty list.

    substitutions : dict (optional)
        This dictionary will be substituted into the problem before solving,
        and also allows the declaration of sweeps and linked sweeps.

    *args, **kwargs : Passed to the setup method for inheritance.
    """
    model_nums = defaultdict(int)

    def __init__(self, cost=None, constraints=None, substitutions=None,
                 *args, **kwargs):
        if cost is None:
            if not hasattr(self, "setup"):
                raise TypeError("Models can only be created without a cost"
                                " if they have a 'setup' method.")
            try:
                name = kwargs.pop("name", self.__class__.__name__)
                setup = self.setup(*args, **kwargs)
            except:
                print("The 'setup' method of this model had an error.")
                raise
            try:
                cost, constraints = setup
            except TypeError:
                raise TypeError("Model 'setup' methods must return "
                                "(cost, constraints).")
        if constraints is None:
            constraints = []
        if substitutions is None:
            substitutions = {}

        self.cost = Signomial(cost)
        self.constraints = list(constraints)
        self.substitutions = dict(substitutions)

        if hasattr(self, "setup"):
            # TODO: use super instead of Model?
            k = Model.model_nums[name]
            Model.model_nums[name] = k+1
            name += str(k) if k else ""
            for s in self.signomials:
                for k in s.varlocs:
                    if "model" not in k.descr:
                        newk = VarKey(k, model=name)
                        s.varlocs[newk] = s.varlocs.pop(k)
                for exp in p.exps:
                    for k in exp:
                        if "model" not in k.descr:
                            newk = VarKey(k, model=name)
                            exp[newk] = exp.pop(k)

    @property
    def allsubs(self):
        subs = {var: var.descr["value"]
                for var in self.variables if "value" in var.descr}
        subs.update(self.substitutions)
        return subs

    @property
    def variables(self):
        variables = {}
        for s in self.signomials:
            variables.update({vk: Variable(**vk.descr) for vk in s.varlocs})
        return variables

    @property
    def signomials(self):
        constraints = tuple(flatten(self.constraints, Constraint))
        # TODO: parse constraints during flattening, calling Posyarray on
        #       anything that holds only posys and then saving that list.
        #       This will allow prettier constraint printing.
        posynomials = [self.cost]
        for constraint in constraints:
            if isinstance(constraint, MonoEQConstraint):
                posynomials.extend([constraint.leq, constraint.geq])
            else:
                posynomials.append(constraint)
        return posynomials

    # TODO: replcae the below with a dynamically created NomialData 'unsubbed'
    @property
    def unsubbed_cs(self):
        return np.hstack((mag(s.cs) for s in self.signomials))

    @property
    def unsubbed_exps(self):
     return functools_reduce(add, (s.exps for s in self.signomials))

    @property
    def unsubbed_varlocs(self):
        return locate_vars(self.unsubbed_exps)[0]

    @property
    def unsubbed_varkeys(self):
        return locate_vars(self.unsubbed_exps)[1]

    @property
    def separate_subs(self):
        "Seperates sweep substitutions from constants."
        # TODO: refactor this
        substitutions = self.allsubs
        varlocs, varkeys = self.unsubbed_varlocs, self.unsubbed_varkeys
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
        """Forms a GeometricProgram and attempts to solve it.

        Arguments
        ---------
        solver : string or function (optional)
            If None, uses the default solver found in installation.

        verbosity : int (optional)
            If greater than 0 prints runtime messages.
            Is decremented by one and then passed to programs.

        skipfailures : bool (optional)
            If True, when a solve errors during a sweep, skip it.

        *args, **kwargs : Passed to solver

        Returns
        -------
        sol : SolutionArray
            See the SolutionArray documentation for details.

        Raises
        ------
        ValueError if called on a model with Signomials.
        RuntimeWarning if an error occurs in solving or parsing the solution.
        """
        try:
            return self._solve("gp", solver, verbosity, skipfailures, *args, **kwargs)
        except ValueError:
            raise ValueError("'solve()' can only be called on models that do"
                             " not contain Signomials, because only those"
                             " models guarantee a global solution."
                             " For a local solution, try 'localsolve()'.")

    def localsolve(self, solver=None, verbosity=2, skipfailures=True, *args, **kwargs):
        """Forms a SignomialProgram and attempts to locally solve it.

        Arguments
        ---------
        solver : string or function (optional)
            If None, uses the default solver found in installation.

        verbosity : int (optional)
            If greater than 0 prints runtime messages.
            Is decremented by one and then passed to programs.

        skipfailures : bool (optional)
            If True, when a solve errors during a sweep, skip it.

        *args, **kwargs : Passed to solver

        Returns
        -------
        sol : SolutionArray
            See the SolutionArray documentation for details.

        Raises
        ------
        ValueError if called on a model without Signomials.
        RuntimeWarning if an error occurs in solving or parsing the solution.
        """
        try:
            return self._solve("sp", solver, verbosity, skipfailures, *args, **kwargs)
        except ValueError:
            raise ValueError("'localsolve()' can only be called on models that"
                             " contain Signomials, because such"
                             " models have only local solutions. Models"
                             " without Signomials have global solutions,"
                             " so try using 'solve()'.")

    def _solve(self, programType, solver, verbosity, skipfailures, *args, **kwargs):
        """Generates a program and solves it, sweeping as appropriate.

        Arguments
        ---------
        solver : string or function (optional)
            If None, uses the default solver found in installation.

        programType : "gp" or "sp"

        verbosity : int (optional)
            If greater than 0 prints runtime messages.
            Is decremented by one and then passed to programs.

        skipfailures : bool (optional)
            If True, when a solve errors during a sweep, skip it.

        *args, **kwargs : Passed to solver

        Returns
        -------
        sol : SolutionArray
            See the SolutionArray documentation for details.

        Raises
        ------
        ValueError if programType and model constraints don't match.
        RuntimeWarning if an error occurs in solving or parsing the solution.
        """
        posynomials = self.signomials
        sweep, linkedsweep, constants = self.separate_subs
        solution = SolutionArray()
        kwargs.update({"solver": solver})
        kwargs.update({"verbosity": verbosity - 1})

        if sweep:
            if len(sweep) == 1:
                sweep_grids = np.array(sweep.values())
            else:
                sweep_grids = np.meshgrid(*list(sweep.values()))

            N_passes = sweep_grids[0].size
            sweep_vects = {var: grid.reshape(N_passes)
                           for (var, grid) in zip(sweep, sweep_grids)}

            if verbosity > 0:
                print("Solving for %i variables over %i passes." %
                      (len(self.variables), N_passes))

            def solve_pass(i):
                this_pass = {var: sweep_vect[i]
                             for (var, sweep_vect) in sweep_vects.items()}
                linked = {var: fn(*[this_pass[VarKey(v)]
                                    for v in var.descr["args"]])
                          for var, fn in linkedsweep.items()}
                this_pass.update(linked)
                constants_ = constants
                constants_.update(this_pass)
                program, mmaps = self.formProgram(programType, posynomials,
                                                  constants_, verbosity)

                try:
                    if programType == "gp":
                        result = program.solve(*args, **kwargs)
                    elif programType == "sp":
                        result = program.localsolve(*args, **kwargs)
                    sol = self.parse_result(result, constants_, mmaps,
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
            self.program, mmaps = self.formProgram(programType, posynomials,
                                                   constants, verbosity)
            if programType == "gp":
                result = self.program.solve(*args, **kwargs)
            elif programType == "sp":
                result = self.program.localsolve(*args, **kwargs)
            sol = self.parse_result(result, constants, mmaps)
            solution.append(sol)

        solution.program = self.program
        solution.toarray()
        self.solution = solution
        return solution

    def gp(self, verbosity=2):
        m, _ = self.formProgram("gp", self.signomials, self.allsubs, verbosity)
        return m

    def sp(self, verbosity=2):
        m, _ = self.formProgram("sp", self.signomials, self.allsubs, verbosity)
        return m

    def formProgram(self, programType, signomials, subs, verbosity=2):
        """Generates a program and solves it, sweeping as appropriate.

        Arguments
        ---------
        programType : "gp" or "sp"

        signomials : list of Signomials
            The first Signomial is the cost function.

        subs : dict
            Substitutions to do before solving.

        verbosity : int (optional)
            If greater than 0 prints runtime messages.
            Is decremented by one and then passed to program inits.

        Returns
        -------
        program : GP or SP
        mmaps : Map from initial monomials to substitued and simplified one.
                See small_scripts.sort_and_simplify for more details.

        Raises
        ------
        ValueError if programType and model constraints don't match.
        """
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
            return GeometricProgram(cost, constraints, verbosity-1), mmaps
        elif programType in ["sp", "SP"]:
            return SignomialProgram(cost, constraints), mmaps
        else:
            raise ValueError("unknown program type %s." % programType)

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, str(self))

    def __str__(self):
        """String representation of a Model.
        Contains all of its parameters."""
        return "\n".join(["# minimize",
                          "    %s," % self.cost,
                          "[   # subject to"] +
                         ["    %s," % constr
                          for constr in self.constraints] +
                         ['],',
                          "    substitutions={ %s }" %
                          pformat(self.allsubs, indent=20)[20:-1]])

    def _latex(self, unused=None):
        """LaTeX representation of a GeometricProgram.
        Contains all of its parameters."""
        sweep, linkedsweep, constants = self.separate_subs
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

    def parse_result(self, result, constants, mmaps, sweep={}, linkedsweep={},
                     freevar_sensitivity_tolerance=1e-4,
                     localmodel_sensitivity_requirement=0.1):
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
            if var in freevariables and abs(S) > freevar_sensitivity_tolerance:
                print("free variable too sensitive:"
                      " S_{%s} = %0.2e" % (var, S))

        localexp = {var: S for (var, S) in sens_vars.items()
                    if abs(S) >= localmodel_sensitivity_requirement}
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

        # TODO: remove after issue #269 is resolved
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
