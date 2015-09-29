# -*- coding: utf-8 -*-
"""Module for creating Model instances.

    Example
    -------
    >>> gp = gpkit.Model(cost, constraints, substitutions)

"""

import numpy as np

from collections import defaultdict

from .nomials import MonoEQConstraint
from .nomials import Signomial
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .solution_array import SolutionArray
from .varkey import VarKey
from . import SignomialsEnabled

from .nomial_data import NomialData

from .solution_array import parse_result
from .substitution import get_constants, separate_subs
from .substitution import substitution
from .small_scripts import flatten, latex_num
from .nomial_data import simplify_exps_and_cs
from .feasibility import feasibility_model

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

    Attributes with side effects
    ----------------------------
    `program` is set during a solve
    `solution` is set at the end of a solve
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
                for exp in s.exps:
                    for k in exp:
                        if "model" not in k.descr:
                            newk = VarKey(k, model=name)
                            exp[newk] = exp.pop(k)

    @property
    def signomials(self):
        constraints = tuple(flatten(self.constraints, Signomial))
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

    @property
    def beforesubs(self):
        "Get this Model's NomialData before any substitutuions"
        return NomialData(nomials=self.signomials)

    @property
    def allsubs(self):
        "All substitutions currently in the Model."
        subs = self.beforesubs.values
        subs.update(self.substitutions)
        return subs

    @property
    def signomials_et_al(self):
        "Get signomials, beforesubs, allsubs in one pass."
        signomials = self.signomials
        beforesubs = NomialData(nomials=signomials)
        allsubs = beforesubs.values
        allsubs.update(self.substitutions)
        return signomials, beforesubs, allsubs

    @property
    def constants(self):
        "All constants (non-sweep substitutions) currently in the Model."
        _, beforesubs, allsubs = self.signomials_et_al
        return get_constants(beforesubs, allsubs)

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

    def solve(self, solver=None, verbosity=2, skipfailures=True,
              *args, **kwargs):
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
            return self._solve("gp", solver, verbosity, skipfailures,
                               *args, **kwargs)
        except ValueError as err:
            if err.message == ("GeometricPrograms cannot contain Signomials"):
                raise ValueError("""Signomials remained after substitution.

    'Model.solve()' can only be called on Models without Signomials, because
    only those Models guarantee a global solution. Models with Signomials
    have only local solutions, and are solved with 'Model.localsolve()'.""")
            raise

    def localsolve(self, solver=None, verbosity=2, skipfailures=True,
                   *args, **kwargs):
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
            with SignomialsEnabled():
                return self._solve("sp", solver, verbosity, skipfailures,
                                   *args, **kwargs)
        except ValueError as err:
            if err.message == ("SignomialPrograms must contain at least one"
                               " Signomial."):
                raise ValueError("""No Signomials remained after substitution.

    'Model.localsolve()' can only be called on models with Signomials,
    since such models have only local solutions. Models without Signomials have
    global solutions, and can be solved with 'Model.solve()'.""")
            raise

    def _solve(self, programType, solver, verbosity, skipfailures,
               *args, **kwargs):
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
        signomials, beforesubs, allsubs = self.signomials_et_al
        beforesubs.signomials = signomials
        sweep, linkedsweep, constants = separate_subs(beforesubs, allsubs)
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

            if verbosity > 1:
                print("Solving over %i passes." % N_passes)

            def solve_pass(i):
                this_pass = {var: sweep_vect[i]
                             for (var, sweep_vect) in sweep_vects.items()}
                linked = {var: fn(*[this_pass[VarKey(v)]
                                    for v in var.descr["args"]])
                          for var, fn in linkedsweep.items()}
                this_pass.update(linked)
                constants_ = constants
                constants_.update(this_pass)
                signomials_, beforesubs.smaps = simplify_and_mmap(signomials,
                                                                constants_)
                program, solvefn = form_program(programType, signomials_,
                                                verbosity=verbosity-1)
                try:
                    result = solvefn(*args, **kwargs)
                    sol = parse_result(result, constants_, beforesubs,
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
                self.program.append(program)  # NOTE: SIDE EFFECTS
                if not hasattr(result, "status"):  # solve succeeded
                    solution.append(result)
                elif not skipfailures:
                    raise RuntimeWarning("solve failed during sweep; program"
                                         " has been saved to m.program[-1]."
                                         " To ignore such failures, solve with"
                                         " skipfailures=True.")
        else:
            signomials, beforesubs.smaps = simplify_and_mmap(signomials,
                                                           constants)
            # NOTE: SIDE EFFECTS
            self.program, solvefn = form_program(programType, signomials,
                                                 verbosity=verbosity-1)
            result = solvefn(*args, **kwargs)
            solution.append(parse_result(result, constants, beforesubs))
        solution.program = self.program
        solution.toarray()
        self.solution = solution  # NOTE: SIDE EFFECTS
        if verbosity > 0:
            print(solution.table())
        return solution

    # TODO: add sweepgp(index)?

    def gp(self, verbosity=2):
        signomials, _ = simplify_and_mmap(self.signomials, self.constants)
        gp, _ = form_program("gp", signomials, verbosity)
        return gp

    def sp(self, verbosity=2):
        signomials, _ = simplify_and_mmap(self.signomials, self.constants)
        sp, _ = form_program("sp", signomials, verbosity)
        return sp

    def feasibility(self,
                    search=["overall", "constraints", "constants"],
                    constvars=None, verbosity=0):
        """Searches for feasibile versions of the Model.

        Argument
        --------
        search : list of strings or string
            The search(es) to perform. Details on each type below.
        constvars : iterable
            If declared, only constants in constvars will be changed.
            Otherwise, all constants can be changed in a constants search.
        verbosity : int
            If greater than 0, will print a report.
            Decremented by 1 and passed to solvers.

        Returns
        -------
        feasibilities : dict, float, or list
            Has an entry for each search; if only one, returns that directly.

            "overall" : float
                The smallest number each constraint's less-than side would
                have to be divided by to make the program feasible.
            "constraints" : array of floats
                Similar to "overall", but contains a number for each
                constraint, and minimizes the product of those numbers.
            "constants" : dict of varkeys: floats
                A substitution dictionary that would make the program feasible,
                chosen to minimize the product of new_values/old_values.

        Examples
        -------
        >>> from gpkit import Variable, Model, PosyArray
        >>> x = Variable("x")
        >>> x_min = Variable("x_min", 2)
        >>> x_max = Variable("x_max", 1)
        >>> m = Model(x, [x <= x_max, x >= x_min])
        >>> # m.solve()  # RuntimeWarning!
        >>> feas = m.feasibility()
        >>>
        >>> # USING OVERALL
        >>> m.constraints = PosyArray(m.signomials)/feas["overall"]
        >>> m.solve()
        >>>
        >>> # USING CONSTRAINTS
        >>> m = Model(x, [x <= x_max, x >= x_min])
        >>> m.constraints = PosyArray(m.signomials)/feas["constraints"]
        >>> m.solve()
        >>>
        >>> # USING CONSTANTS
        >>> m = Model(x, [x <= x_max, x >= x_min])
        >>> m.substitutions.update(feas["constants"])
        >>> m.solve()
        """
        signomials, unsubbed, allsubs = self.signomials_et_al

        try:
            self.gp()
        except ValueError as err:
            if err.message == ("GeometricPrograms cannot contain Signomials"):
                raise ValueError("""Signomials remained after substitution.

    'Model.feasibility()' can only be called on Models without Signomials,
    because only those Models guarantee global feasibilities. Models with
    Signomials have only local feasibilities, which can be found with
    'Model.localfeasibility()'.""")
            raise
        else:
            return self._feasibility("gp", search, constvars, verbosity)

    def localfeasibility(self,
                         search=["overall", "constraints", "constants"],
                         constvars=None, verbosity=0):
        """Searches for locally feasibile versions of the Model.

        For details, see the docstring for Model.feasibility.
        """
        try:
            self.sp()
        except ValueError as err:
            if err.message == ("SignomialPrograms must contain at least one"
                               " Signomial."):
                raise ValueError("""No Signomials remained after substitution.

    'Model.localfeasibility()' can only be called on models containing
    Signomials, since such models have only local feasibilities. Models without
    Signomials have global feasibilities, which can be found with
    'Model.feasibility()'.""")
            raise
        else:
            return self._feasibility("sp", search, constvars, verbosity)

    def _feasibility(self, programtype, search, constvars, verbosity):
        signomials, unsubbed, allsubs = self.signomials_et_al
        feasibilities = {}

        if "overall" in search:
            m = feasibility_model(self, "max")
            m.substitutions = allsubs
            infeasibility = m._solve(programtype, None, verbosity-1, False)["cost"]
            feasibilities["overall"] = infeasibility

        if "constraints" in search:
            m = feasibility_model(self, "product")
            m.substitutions = allsubs
            sol = m._solve(programtype, None, verbosity-1, False)
            feasibilities["constraints"] = sol(m.slackvars)

        if "constants" in search:
            constants = get_constants(unsubbed, allsubs)
            if constvars:
                constvars = set(constvars)
                # get varkey versions
                constvars = get_constants(unsubbed.varkeys, unsubbed.varlocs,
                                          dict(zip(constvars, constvars)))
                # filter constants
                constants = {k: v for k, v in constants.items()
                             if k in constvars}
            if constants:
                m = feasibility_model(self, "constants", constants=constants)
                sol = m._solve(programtype, None, verbosity-1, False)
                feasiblevalues = sol(m.constvars).tolist()
                changed_vals = feasiblevalues != np.array(m.constvalues)
                var_infeas = {m.addvalue[m.constvarkeys[i]]: feasiblevalues[i]
                              for i in np.where(changed_vals)}
                feasibilities["constants"] = var_infeas

        if len(feasibilities) > 1:
            return feasibilities
        else:
            return feasibilities.values()[0]

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
                          "    substitutions=%s" % self.allsubs])

    def _latex(self, unused=None):
        """LaTeX representation of a GeometricProgram.
        Contains all of its parameters."""
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
                                 for var, val in self.constants.items()]) +
                         ["\\end{array}"])


def form_program(programType, signomials, verbosity=2):
    "Generates a program and returns it and its solve function."
    cost, constraints = signomials[0], signomials[1:]
    if programType == "gp":
        gp = GeometricProgram(cost, constraints, verbosity)
        return gp, gp.solve
    elif programType == "sp":
        sp = SignomialProgram(cost, constraints, verbosity)
        return sp, sp.localsolve
    else:
        raise ValueError("unknown program type %s." % programType)


def simplify_and_mmap(constraints, subs):
    """Simplifies a list of constraints and returns them with their mmaps.

    Arguments
    ---------
    constraints : list of Signomials
    subs : dict
        Substitutions to do before simplifying.

    Returns
    -------
    constraints : list of simplified Signomials
        Signomials with cs that are solely nans and/or zeroes are removed.

    mmaps : Map from initial monomials to substitued and simplified one.
            See small_scripts.sort_and_simplify for more details.
    """
    signomials_, smaps = [], []
    for s in constraints:
        _, exps, cs, _ = substitution(s, subs)
        # remove any cs that are just nans and/or 0s
        notnan = ~np.isnan(cs)
        if np.any(notnan) and np.any(cs[notnan] != 0):
            exps, cs, smap = simplify_exps_and_cs(exps, cs, return_map=True)
            if s is not constraints[0] and s.any_nonpositive_cs:
                # if s is still a Signomial cost we'll let SP throw the error
                # if s was a Signomial constraint, catch impossibilities
                # and convert to Posynomial consrtaints as possible
                negative_c_count = (cs <= 0).sum()
                if negative_c_count == 0:
                    raise RuntimeWarning("""Infeasible SignomialConstraint.

    %s became infeasible  when all negative terms were substituted out.""" % s)
                elif negative_c_count == 1:
                    # turn it into a Posynomial constraint
                    idx = cs.argmin()
                    exps = list(exps)
                    div_exp = exps.pop(idx)
                    div_mmap = smap.pop(idx)
                    cs /= -cs[idx]
                    cs = np.hstack((cs[:idx], cs[idx+1:]))
                    exps = tuple(exp-div_exp for exp in exps)
                    smap = [mmap-div_mmap for mmap in smap]
            signomials_.append(Signomial(exps, cs, simplify=False))
            smaps.append(smap)
        else:
            # This constraint is being removed; append an empty smap so that
            # smaps keeps the same length as signomials.
            smaps.append([])

    return signomials_, smaps
