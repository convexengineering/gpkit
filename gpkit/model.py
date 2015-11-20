# -*- coding: utf-8 -*-
"""Module for creating Model instances.

    Example
    -------
    >>> gp = gpkit.Model(cost, constraints, substitutions)

"""

import numpy as np

from collections import defaultdict
from time import time

from .small_classes import Numbers, Strings
from .nomials import MonoEQConstraint
from .nomials import Signomial, Monomial
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .solution_array import SolutionArray
from .varkey import VarKey
from .variables import Variable, VectorVariable
from .posyarray import PosyArray
from . import SignomialsEnabled

from .nomial_data import NomialData

from .solution_array import parse_result
from .substitution import get_constants, separate_subs
from .substitution import substitution
from .small_scripts import mag, flatten, latex_num
from .nomial_data import simplify_exps_and_cs
from .feasibility import feasibility_model

try:
    from ipyparallel import Client
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
        if substitutions is None:
            substitutions = {}
        isobjectmodel = hasattr(self, "setup")  # not sure about the name
        if isobjectmodel:
            try:
                substitutionsarg = substitutions
                extended_args = [arg for arg in [cost, constraints]
                                 if arg is not None]
                extended_args.extend(args)
                name = kwargs.pop("name", self.__class__.__name__)
                setup = self.setup(*extended_args, **kwargs)
            except:
                print("The 'setup' method of this model had an error.")
                raise
            try:
                if isinstance(setup, Model):
                    cost, constraints = setup.cost, setup.constraints
                    substitutions = setup.substitutions
                elif len(setup) == 2:
                    cost, constraints = setup
                    substitutions = {}
                elif len(setup) == 3:
                    cost, constraints, substitutions = setup
                substitutions.update(substitutionsarg)
            except TypeError:
                raise TypeError("Model 'setup' methods must return either"
                                " a Model, a tuple of (cost, constraints),"
                                " or a tuple of (cost, constraints,"
                                " substitutions).")

        self.cost = Signomial(cost) if cost else Monomial(1)
        self.constraints = list(constraints) if constraints else []
        self.substitutions = substitutions
        for key, value in substitutions.items():
            # update string keys to varkeys
            if isinstance(key, Strings):
                del self.substitutions[key]
                self.substitutions[self[key]] = value

        if isobjectmodel:
            # TODO: use super instead of Model?
            k = Model.model_nums[name]
            Model.model_nums[name] = k+1
            name += str(k) if k else ""
            processed_keys = set()
            for s in self.signomials:
                for k in s.varlocs:
                    if k not in processed_keys:
                        processed_keys.add(k)
                        model = name+k.descr.get("model", "")
                        k.descr["model"] = model
                for exp in s.exps:
                    for k in exp:
                        if k not in processed_keys:
                            processed_keys.add(k)
                            model = name + k.descr.pop("model", "")
                            k.descr["model"] = model
            for k, v in self.substitutions.items():
                # doesn't work for Var / Vec substitution yet
                kmodel = name + k.descr.pop("model", "")
                if k not in processed_keys:
                    processed_keys.add(k)
                    k.descr["model"] = kmodel
                if isinstance(v, VarKey):
                    vmodel = name + v.descr.pop("model", "")
                    if v not in processed_keys:
                        processed_keys.add(v)
                        v.descr["model"] = vmodel

    def __or__(self, other):
        return self.concat(other)

    def __and__(self, other):
        return self.merge(other)

    def __getitem__(self, item):
        # note: this rebuilds the dictionary on every acess
        # if this is too slow, there could be some hashing and caching
        return self.varsbyname[item]

    def merge(self, other, excluded_names=[]):
        if not isinstance(other, Model):
            return NotImplemented
        selfvars = self.varsbyname
        othervars = other.varsbyname
        overlap = set(selfvars) & set(othervars)
        substitutions = dict(self.substitutions)
        substitutions.update(other.substitutions)
        for name in overlap:
            if name in excluded_names:
                continue
            descr = self[name].key.descr
            descr.pop("model", None)
            newvar = VarKey(**descr)
            svars = (selfvars[name] if isinstance(selfvars[name], list)
                     else [selfvars[name]])
            ovars = (othervars[name] if isinstance(othervars[name], list)
                     else [othervars[name]])
            for var in svars + ovars:
                if var.key != newvar.key:
                    substitutions[var.key] = newvar.key
        return Model(self.cost,
                     self.constraints + other.constraints,
                     substitutions)

    def concat(self, other):
        if not isinstance(other, Model):
            return NotImplemented
        substitutions = dict(self.substitutions)
        substitutions.update(other.substitutions)
        return Model(self.cost,
                     self.constraints + other.constraints,
                     substitutions)

    @property
    def varkeys(self):
        varkeys = set()
        for signomial in self.signomials:
            for exp in signomial.exps:
                varkeys.update(exp)
        return {str(vk): vk for vk in varkeys}

    @property
    def varsbyname(self):
        varsbyname = defaultdict(list)
        for varkey in self.varkeys.values():
            if varkey in self.substitutions:
                sub = self.substitutions[varkey]
                if isinstance(sub, VarKey):
                    varkey = sub
            var = Variable(**varkey.descr)
            if "idx" in varkey.descr and "shape" in varkey.descr:
                descr = dict(varkey.descr)
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
                    vecidx = [i for i, li in enumerate(varsbyname[varkey.name])
                              if veckey == li][0]
                    varsbyname[varkey.name][vecidx][idx] = var
                except IndexError:  # veckey not in list!
                    nanarray = np.full(var.descr["shape"], np.nan, dtype="object")
                    nanPosyArray = PosyArray(nanarray)
                    nanPosyArray[idx] = var
                    nanPosyArray.key = veckey
                    varsbyname[varkey.name].append(nanPosyArray)
            else:
                if var not in varsbyname[varkey.name]:
                    varsbyname[varkey.name].append(var)
        for name, variables in varsbyname.items():
            if len(variables) == 1:
                varsbyname[name] = variables[0]
        return dict(varsbyname)

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
        return NomialData.fromnomials(self.signomials)

    @property
    def allsubs(self):
        "All substitutions currently in the Model."
        subs = self.beforesubs.values
        subs.update(self.substitutions)
        return subs

    @property
    def signomials_et_al(self):
        "Get signomials, beforesubs, allsubs in one pass; applies VarKey subs."
        signomials = self.signomials
        # don't use self.beforesubs here to avoid re-computing self.signomials
        beforesubs = NomialData.fromnomials(signomials)
        allsubs = beforesubs.values
        allsubs.update(self.substitutions)
        varkeysubs = {vk: nvk for vk, nvk in allsubs.items()
                      if isinstance(nvk, VarKey)}
        if varkeysubs:
            beforesubs = beforesubs.sub(varkeysubs, require_positive=False)
            beforesubs.varkeysubs = varkeysubs
            signomials = [s.sub(varkeysubs, require_positive=False)
                          for s in signomials]
        return signomials, beforesubs, allsubs

    @property
    def constants(self):
        "All constants (non-sweep substitutions) currently in the Model."
        _, beforesubs, allsubs = self.signomials_et_al
        return get_constants(beforesubs, allsubs)

    def zero_lower_unbounded_variables(self):
        "Recursively substitutes 0 for variables that lack a lower bound"
        zeros = True
        while zeros:
            bounds = self.gp(verbosity=0).missingbounds
            zeros = {var: 0 for var, bound in bounds.items()
                     if bound == "lower"}
            self.substitutions.update(zeros)

    def solve(self, solver=None, verbosity=2, skipsweepfailures=False,
              *args, **kwargs):
        """Forms a GeometricProgram and attempts to solve it.

        Arguments
        ---------
        solver : string or function (optional)
            If None, uses the default solver found in installation.

        verbosity : int (optional)
            If greater than 0 prints runtime messages.
            Is decremented by one and then passed to programs.

        skipsweepfailures : bool (optional)
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
            return self._solve("gp", solver, verbosity, skipsweepfailures,
                               *args, **kwargs)
        except ValueError as err:
            if err.message == ("GeometricPrograms cannot contain Signomials"):
                raise ValueError("""Signomials remained after substitution.

    'Model.solve()' can only be called on Models without Signomials, because
    only those Models guarantee a global solution. Models with Signomials
    have only local solutions, and are solved with 'Model.localsolve()'.""")
            raise

    def localsolve(self, solver=None, verbosity=2, skipsweepfailures=False,
                   *args, **kwargs):
        """Forms a SignomialProgram and attempts to locally solve it.

        Arguments
        ---------
        solver : string or function (optional)
            If None, uses the default solver found in installation.

        verbosity : int (optional)
            If greater than 0 prints runtime messages.
            Is decremented by one and then passed to programs.

        skipsweepfailures : bool (optional)
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
                return self._solve("sp", solver, verbosity, skipsweepfailures,
                                   *args, **kwargs)
        except ValueError as err:
            if err.message == ("SignomialPrograms must contain at least one"
                               " Signomial."):
                raise ValueError("""No Signomials remained after substitution.

    'Model.localsolve()' can only be called on models with Signomials,
    since such models have only local solutions. Models without Signomials have
    global solutions, and can be solved with 'Model.solve()'.""")
            raise

    def _solve(self, programType, solver, verbosity, skipsweepfailures,
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

        skipsweepfailures : bool (optional)
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
        if any(isinstance(val, Numbers) and val == 0
               for val in self.allsubs.values()):
            if verbosity > 1:
                print("A zero-substitution triggered the zeroing of lower-"
                      "unbounded variables to maintain solver compatibility.")
            self.zero_lower_unbounded_variables()
        signomials, beforesubs, allsubs = self.signomials_et_al
        beforesubs.signomials = signomials
        sweep, linkedsweep, constants = separate_subs(beforesubs, allsubs)
        solution = SolutionArray()
        kwargs.update({"solver": solver})

        if sweep:
            kwargs.update({"verbosity": verbosity-2})
            if len(sweep) == 1:
                sweep_grids = np.array(list(sweep.values()))
            else:
                sweep_grids = np.meshgrid(*list(sweep.values()))

            N_passes = sweep_grids[0].size
            sweep_vects = {var: grid.reshape(N_passes)
                           for (var, grid) in zip(sweep, sweep_grids)}

            if verbosity > 1:
                print("Solving over %i passes." % N_passes)
                tic = time()

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
                                                verbosity=verbosity)
                try:
                    result = solvefn(*args, **kwargs)
                    sol = parse_result(result, constants_, beforesubs,
                                       sweep, linkedsweep)
                    return program, sol
                except (RuntimeWarning, ValueError):
                    return program, None

            if POOL:
                mapfn = POOL.map_sync
            else:
                mapfn = map

            self.program = []
            for program, result in mapfn(solve_pass, range(N_passes)):
                self.program.append(program)  # NOTE: SIDE EFFECTS
                if result:  # solve succeeded
                    solution.append(result)
                elif not skipsweepfailures:
                    raise RuntimeWarning("solve failed during sweep; program"
                                         " has been saved to m.program[-1]."
                                         " To ignore such failures, solve with"
                                         " skipsweepfailures=True.")
            for var, val in solution["constants"].items():
                solution["constants"][var] = [val[0]]

            if verbosity > 1:
                soltime = time() - tic
                print("Sweeping took %.3g seconds." % (soltime,))
        else:
            kwargs.update({"verbosity": verbosity-1})
            signomials, beforesubs.smaps = simplify_and_mmap(signomials,
                                                             constants)
            # NOTE: SIDE EFFECTS
            self.program, solvefn = form_program(programType, signomials,
                                                 verbosity=verbosity)
            result = solvefn(*args, **kwargs)
            solution.append(parse_result(result, constants, beforesubs))
        solution.program = self.program
        solution.toarray()
        solution["localmodel"] = PosyArray(solution["localmodel"])
        self.solution = solution  # NOTE: SIDE EFFECTS
        if verbosity > 0:
            print(solution.table())
        return solution

    # TODO: add sweepgp(index)?

    def gp(self, verbosity=2):
        signomials, _ = simplify_and_mmap(self.signomials, self.constants)
        gp, _ = form_program("gp", signomials, verbosity)
        return gp

    @property
    def isGP(self):
        try:
            self.gp()
            return True
        except ValueError as err:
            if err.message == ("GeometricPrograms cannot contain Signomials"):
                return False
            else:
                raise err

    @property
    def isSP(self):
        return not self.isGP

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

        if self.isSP:
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
        if self.isGP:
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
            infeasibility = m._solve(programtype, None, verbosity, False)["cost"]
            feasibilities["overall"] = infeasibility

        if "constraints" in search:
            m = feasibility_model(self, "product")
            m.substitutions = allsubs
            sol = m._solve(programtype, None, verbosity, False)
            feasibilities["constraints"] = sol(m.slackvars)

        if "constants" in search:
            constants = get_constants(unsubbed, allsubs)
            signomials, _ = simplify_and_mmap(signomials, {})
            if constvars:
                constvars = set(constvars)
                # get varkey versions
                constvars = get_constants(unsubbed.varkeys, unsubbed.varlocs,
                                          dict(zip(constvars, constvars)))
                # filter constants
                constants = {k: v for k, v in constants.items()
                             if k in constvars}
            if constants:
                m = feasibility_model(self, "constants", constants=constants,
                                      signomials=signomials, programType=Model)
                sol = m.solve(verbosity=verbosity)
                feasiblevalues = sol(m.constvars)
                var_infeas = {}
                for i, slackval in enumerate(sol(m.slackb)):
                    if slackval > 1.01:
                        original_varkey = m.addvalue[m.constvarkeys[i]]
                        var_infeas[original_varkey] = feasiblevalues[i]
                feasibilities["constants"] = var_infeas

        if len(feasibilities) == 1:
            return feasibilities.values()[0]
        else:
            return feasibilities

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

    def latex(self, show_subs=True):
        """LaTeX representation of a GeometricProgram.
        Contains all of its parameters."""
        # TODO: print sweeps and linkedsweeps
        latex_list = ["\\begin{array}{ll}",
                      "\\text{}",
                      "\\text{minimize}",
                      "    & %s \\\\" % self.cost.latex(),
                      "\\text{subject to}"]
        latex_list += ["    & %s \\\\" % constr.latex()
                       for constr in self.constraints]
        if show_subs:
            sub_latex = ["    & %s \gets %s%s \\\\" % (var.latex(),
                                                        latex_num(val),
                                                        var.unitstr)
                         for var, val in self.constants.items()]
            latex_list += ["\\text{substituting}"] + sorted(sub_latex)
        latex_list += ["\\end{array}"]
        return "\n".join(latex_list)

    def _repr_latex_(self):
        return "$$"+self.latex()+"$$"

    def interact(self, ranges=None, fn_of_sol=None, **solvekwargs):
        """Easy model interaction in IPython / Jupyter

        By default, this creates a model with sliders for every constant
        which prints a new solution table whenever the sliders are changed.

        Arguments
        ---------
        fn_of_sol : function
            The function called with the solution after each solve that
            displays the result. By default prints a table.

        ranges : dictionary {str: Slider object or tuple}
            Determines which sliders get created. Tuple values may contain
            two or three floats: two correspond to (min, max), while three
            correspond to (min, step, max)

        **solvekwargs
            kwargs which get passed to the solve()/localsolve() method.
        """
        from .interactive.widgets import modelinteract
        return modelinteract(self, ranges, fn_of_sol, **solvekwargs)

    def controlpanel(self, *args, **kwargs):
        """Easy model control in IPython / Jupyter

        Like interact(), but with the ability to control sliders and their ranges
        live. args and kwargs are passed on to interact()
        """
        from .interactive.widgets import modelcontrolpanel
        return modelcontrolpanel(self, *args, **kwargs)


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
                negative_c_count = (mag(cs) <= 0).sum()
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
