# -*- coding: utf-8 -*-
"""Module for creating Model instances.

    Example
    -------
    >>> gp = gpkit.Model(cost, constraints, substitutions)

"""

import numpy as np

from collections import defaultdict
from time import time

from .small_classes import Numbers, Strings, KeySet, KeyDict
from .nomials import MonoEQConstraint
from .nomials import PosynomialConstraint, SignomialConstraint
from .nomials import Signomial, Monomial
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .solution_array import SolutionArray
from .varkey import VarKey
from .variables import Variable, VectorVariable
from .posyarray import PosyArray
from . import SignomialsEnabled

from .nomial_data import NomialData

from .substitution import parse_subs
from .substitution import substitution
from .small_scripts import mag, flatten, latex_num, veckeyed
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
        subs = self.cost.values
        subs.update(substitutions)
        self.substitutions = KeyDict.from_constraints(self.varkeys,
                                                      self.constraints, subs)
        self.modelname = None
        if isobjectmodel:
            # TODO: use super instead of Model?
            k = Model.model_nums[name]
            Model.model_nums[name] = k+1
            name += str(k) if k else ""
            self.modelname = name
            processed_keys = set()
            for c in self.constraints:
                for k in c.varkeys:
                    if k not in processed_keys:
                        processed_keys.add(k)
                        model = name + k.descr.get("model", "")
                        k.descr["model"] = model
            for k, v in self.substitutions.items():
                # doesn't work for Var / Vec substitution yet
                if k not in processed_keys:
                    k = k.key
                    processed_keys.add(k)
                    # implement nested model names
                    kmodel = name + k.descr.pop("model", "")
                    k.descr["model"] = kmodel
                if isinstance(v, VarKey):
                    if v not in processed_keys:
                        processed_keys.add(v)
                        vmodel = name + v.descr.pop("model", "")
                        v.descr["model"] = vmodel

    def __or__(self, other):
        return self.concat(other)

    def __and__(self, other):
        return self.merge(other)

    def __getitem__(self, item):
        # note: this rebuilds the dictionary on every acess
        # if this is too slow, there could be some hashing and caching
        return self.varsbyname[item]

    def merge(self, other, excluded=None):
        if not isinstance(other, Model):
            return NotImplemented
        if excluded is None:
            excluded = []
        selfvars = self.varsbyname
        othervars = other.varsbyname
        overlap = set(selfvars) & set(othervars)
        varsubs = {}
        for name in overlap:
            if name in excluded:
                continue
            svars = selfvars[name]
            if not isinstance(svars, list):
                svars = [svars]
            ovars = othervars[name]
            if not isinstance(ovars, list):
                ovars = [ovars]
            # descr is taken from the first varkey of a given name in self.
            # The loop below compares it to other varkeys of that name.
            descr = dict(svars[0].key.descr)
            descr.pop("model", None)
            for var in svars + ovars:
                descr_ = dict(var.key.descr)
                descr_.pop("model", None)
                # if values disagree, drop self's value
                if descr.get("value", None) != descr_.get("value", None):
                    descr.pop("value", None)
            newvar = VarKey(**descr)
            for var in svars + ovars:
                if var.key != newvar.key:
                    varsubs[var.key] = newvar.key
        cost = self.cost.sub(varsubs)
        constraints = [c.sub(varsubs)
                       for c in self.constraints + other.constraints]
        substitutions = dict(self.substitutions)
        substitutions.update(other.substitutions)
        for var, subvar in varsubs.items():
            if var in substitutions:
                substitutions[subvar] = substitutions[var]
                del substitutions[var]
        return Model(self.cost, constraints, substitutions)

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
        varkeys = KeySet(self.cost.varkeys)
        for constraint in self.constraints:
            varkeys.update(constraint.varkeys)
        return varkeys

    @property
    def allsubs(self):
        newsubs = KeyDict.from_constraints(self.varkeys, self.constraints)
        self.substitutions.update(newsubs)
        return self.substitutions

    @property
    def varsbyname(self):
        varsbyname = defaultdict(list)
        for varkey in self.varkeys:
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
        # if any(isinstance(val, Numbers) and val == 0
        #        for val in self.allsubs.values()):
        #     if verbosity > 1:
        #         print("A zero-substitution triggered the zeroing of lower-"
        #               "unbounded variables to maintain solver compatibility.")
        #     self.zero_lower_unbounded_variables()

        constants, sweep, linkedsweep = parse_subs(self.varkeys, self.allsubs)
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
                constants.update(this_pass)
                program, solvefn = self.form_program(programType, verbosity,
                                                     substitutions=constants)
                try:
                    result = solvefn(*args, **kwargs)
                    return program, result
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

            solution["sweepvariables"] = KeyDict()
            ksweep, klinkedsweep = KeyDict(sweep), KeyDict(linkedsweep)
            for var, val in solution["constants"].items():
                if var in ksweep or var in klinkedsweep:
                    solution["sweepvariables"][var] = val
                    del solution["constants"][var]
                else:
                    solution["constants"][var] = [val[0]]
            if not solution["constants"]:
                del solution["constants"]

            if verbosity > 1:
                soltime = time() - tic
                print("Sweeping took %.3g seconds." % (soltime,))
        else:
            kwargs.update({"verbosity": verbosity-1})
            # NOTE: SIDE EFFECTS
            self.program, solvefn = self.form_program(programType, verbosity,
                                                      substitutions=constants)
            result = solvefn(*args, **kwargs)
            # add localmodel here
            solution.append(result)
        solution.program = self.program
        solution.toarray()
        # solution["localmodel"] = PosyArray(solution["localmodel"])
        self.solution = solution  # NOTE: SIDE EFFECTS
        if verbosity > 0:
            print(solution.table())
        return solution

    # TODO: add sweepgp(index)?

    def gp(self, verbosity=2):
        gp, _ = self.form_program("gp", verbosity)
        return gp

    def sp(self, verbosity=2):
        sp, _ = self.form_program("sp", verbosity)
        return sp

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
        latex_list = ["\\begin{array}[ll]",
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
                         for var, val in self.allsubs.items()]
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

    def form_program(self, programType, verbosity=2, substitutions=None):
        "Generates a program and returns it and its solve function."
        subs = substitutions if substitutions else self.allsubs
        if programType == "gp":
            gp = GeometricProgram(self.cost, self.constraints, subs, verbosity)
            return gp, gp.solve
        elif programType == "sp":
            sp = SignomialProgram(self.cost, self.constraints, subs, verbosity)
            return sp, sp.localsolve
        else:
            raise ValueError("unknown program type %s." % programType)
