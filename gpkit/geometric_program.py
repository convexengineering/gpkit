# -*- coding: utf-8 -*-
"""Module for creating GeometricProgram instances.

    Example
    -------
    >>> gp = gpkit.GeometricProgram(cost, constraints, substitutions)

"""

import numpy as np

from time import time
from pprint import pformat
from collections import defaultdict
import functools
from operator import mul

from .small_classes import Strings
from .small_classes import DictOfLists
from .model import Model
from .nomials import Constraint, MonoEQConstraint
from .nomials import Monomial
from .varkey import VarKey
from .posyarray import PosyArray

from .small_scripts import latex_num
from .small_scripts import flatten
from .small_scripts import locate_vars
from .small_scripts import results_table
from .small_scripts import mag
from .small_scripts import unitstr

try:
    from IPython.parallel import Client
    CLIENT = Client(timeout=0.01)
    assert len(CLIENT) > 0
    POOL = CLIENT[:]
    POOL.use_dill()
    print("Using parallel execution of sweeps on %s clients" % len(CLIENT))
except:
    POOL = None


class GPSolutionArray(DictOfLists):
    "DictofLists extended with posynomial substitution."

    def __len__(self):
        try:
            return len(self["cost"])
        except TypeError:
            return 1

    def __call__(self, p):
        return mag(self.subinto(p).c)

    def getvars(self, *args):
        out = [self["variables"][arg] for arg in args]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def subinto(self, p):
        "Returns PosyArray of each solution substituted into p."
        if p in self["variables"]:
            return PosyArray(self["variables"][p])
        if len(self) > 1:
            return PosyArray([p.sub(self.atindex(i)["variables"])
                              for i in range(len(self["cost"]))])
        else:
            return p.sub(self["variables"])

    def senssubinto(self, p):
        """Returns array of each solution's sensitivity substituted into p

        Returns only scalar values.
        """
        if len(self) > 1:
            subbeds = [p.sub(self.atindex(i)["sensitivities"]["variables"],
                             require_positive=False) for i in range(len(self))]
            assert not any([subbed.exp for subbed in subbeds])
            return np.array([mag(subbed.c) for subbed in subbeds],
                            np.dtype('float'))
        else:
            subbed = p.sub(self["sensitivities"]["variables"],
                           require_positive=False)
            assert isinstance(subbed, Monomial)
            assert not subbed.exp
            return subbed.c

    def table(self,
              tables=["cost", "freevariables", "sweptvariables",
                      "constants", "sensitivities"],
              fixedcols=True):
        if isinstance(tables, Strings):
            tables = [tables]
        strs = []
        if "cost" in tables:
            strs += ["\nCost\n----"]
            if len(self) > 1:
                costs = ["%-8.3g" % c for c in self["cost"][:4]]
                strs += [" [ %s %s ]" % ("  ".join(costs),
                                        "..." if len(self) > 4 else "")]
            else:
                strs += [" %-.4g" % self["cost"]]
            strs[-1] += unitstr(self.gp.cost.units, into=" [%s] ", dimless="")
            strs += [""]
        if "variables" in tables:
            strs += [results_table(self["variables"],
                                   "Variables",
                                   fixedcols=fixedcols)]
        if "freevariables" in tables:
            strs += [results_table({k: v for (k, v) in self["variables"].items()
                                         if k not in self.gp.substitutions
                                         and k not in self.gp.sweep},
                                   "Free variables",
                                   fixedcols=fixedcols)]
        if "sweptvariables" in tables:
            strs += [results_table({k: v for (k, v) in self["variables"].items()
                                         if k in self.gp.sweep},
                                   "Swept variables",
                                   fixedcols=fixedcols)]
        if "constants" in tables:
            strs += [results_table({k: v for (k, v) in self["variables"].items()
                                         if k in self.gp.substitutions},
                                   "Constants",
                                   fixedcols=fixedcols)]
        if "sensitivities" in tables:
            strs += [results_table(self["sensitivities"]["variables"],
                                   "Constant and swept variable sensitivities",
                                   fixedcols=fixedcols,
                                   minval=1e-2,
                                   printunits=False)]
        return "\n".join(strs)


class GeometricProgram(Model):
    """Holds a model and cost function for passing to solvers.

    Arguments
    ---------
    cost : Constraint
        Posynomial to minimize when solving
    constraints : list of (lists of) Constraints
        Constraints to maintain when solving (MonoEQConstraints will
        be turned into <= and >= constraints)
    substitutions : dict {varname: float or int} (optional)
        Substitutions to be applied before solving (including sweeps)
    solver : str (optional)
        Name of solver to use
    options : dict (optional)
        Options to pass to solver

    Examples
    --------
    >>> gp = gpkit.GeometricProgram(  # minimize
                        0.5*rho*S*C_D*V**2,
                        [   # subject to
                            Re <= (rho/mu)*V*(S/A)**0.5,
                            C_f >= 0.074/Re**0.2,
                            W <= 0.5*rho*S*C_L*V**2,
                            W <= 0.5*rho*S*C_Lmax*V_min**2,
                            W >= W_0 + W_w,
                            W_w >= W_w_surf + W_w_strc,
                            C_D >= C_D_fuse + C_D_wpar + C_D_ind
                        ], substitutions)
    >>> gp.solve()

    """

    model_nums = defaultdict(int)

    def __init__(self, *args, **kwargs):
        args = list(args)
        self.result = {}
        if hasattr(self, "setup"):
            # setup cost/constraints replace init ones except for "model"
            if "name" in kwargs:
                name = kwargs.pop("name")
            else:
                name = self.__class__.__name__
            ans = self.setup(*args, **kwargs)
            try:
                self.cost, constraints = ans
            except TypeError:
                raise TypeError("GeometricProgram setup must return "
                                "(cost, constraints).")
        else:
            if "cost" in kwargs:
                self.cost = kwargs["cost"]
            else:
                self.cost = args.pop(0)
            if "constraints" in kwargs:
                constraints = kwargs["constraints"]
            else:
                if args:
                    constraints = args.pop(0)
                else:
                    constraints = []
        self.constraints = tuple(flatten(constraints, Constraint))
        # TODO: parse constraints during flattening, calling Posyarray on
        #       anything that holds only posys and then saving that list.
        #       This will allow prettier constraint printing.
        posynomials = [self.cost]
        for constraint in self.constraints:
            if isinstance(constraint, MonoEQConstraint):
                posynomials += [constraint.leq, constraint.geq]
            else:
                posynomials.append(constraint)
        self.posynomials = tuple(posynomials)
        if hasattr(self, "setup"):
            k = GeometricProgram.model_nums[name]
            GeometricProgram.model_nums[name] = k+1
            name += str(k) if k else ""
            for p in self.posynomials:
                for k in p.varlocs:
                    if "model" not in k.descr:
                        newk = VarKey(k, model=name)
                        p.varlocs[newk] = p.varlocs.pop(k)
                for exp in p.exps:
                    for k in exp:
                        if "model" not in k.descr:
                            newk = VarKey(k, model=name)
                            exp[newk] = exp.pop(k)

        self.sweep, self.linkedsweep = {}, {}
        self._gen_unsubbed_vars()
        values = {var: var.descr["value"]
                  for var in self.varlocs if "value" in var.descr}
        if "substitutions" in kwargs:
            values.update(kwargs["substitutions"])
        elif len(args) > 0 and not hasattr(self, "setup"):
            values.update(args.pop(0))
        if values and not hasattr(self, "setup"):
            self.sub(values)
            self.initalsub = self.last
        else:
            self._gen_unsubbed_vars()

        if "solver" in kwargs:
            self.solver = kwargs["solver"]
        else:
            from . import settings
            self.solver = settings['installed_solvers'][0]
        if "options" in kwargs:
            self.options = kwargs["options"]
        else:
            self.options = {}

    def __add__(self, other):
        if isinstance(other, GeometricProgram):
            # don't add costs b/c that breaks when costs have units
            newgp = GeometricProgram(self.cost * other.cost,
                                     self.constraints + other.constraints)
            subs = newgp.substitutions
            newgp._gen_unsubbed_vars([self.exps[0] + other.exps[0]] +
                                     self.exps[1:] + other.exps[1:],
                                     np.hstack([self.cs[0] * other.cs[0],
                                               self.cs[1:], other.cs[1:]]))
            values = {var: var.descr["value"]
                      for var in newgp.varlocs if "value" in var.descr}
            if values:
                newgp.sub(values)
                newgp.initalsub = newgp.last
                subs.update(values)
                newgp.substitutions = subs
            return newgp
        else:
            return NotImplemented

    def vars(self, *args):
        out = [self[arg] for arg in args]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def __getitem__(self, key):
        for attr in ["result", "solution", "solv", "variables"]:
            solution_variables = (attr == "solv" and hasattr(self, "solution"))
            if hasattr(self, attr) or solution_variables:
                if attr == "solv":
                    d = self.solution["variables"]
                else:
                    d = getattr(self, attr)
                if key in d:
                    return d[key]
                elif key+"_0" in d:
                    maybevec = self.varkeys[key+"_0"].descr
                    if "shape" in maybevec:
                        # then it is a vector!
                        length = maybevec["shape"]
                        l = [d[key+"_%s" % i] for i in range(length)]
                        if attr == "variables":
                            return PosyArray(l)
                        else:
                            return np.array(l)

        raise KeyError("'%s' was not found as a result solution or variable."
                       % key)

    def __setitem__(self, key, value):
        self.results[key] = value

    def __eq__(self, other):
        "GeometricProgram equality is determined by string representations."
        return str(self) == str(other)

    def __ne__(self, other):
        "GeometricProgram inequality is determined by string representations."
        return not self == other

    def __repr__(self):
        """String representation of a GeometricProgram.
        Contains all of its parameters."""
        return "\n".join(["gpkit.GeometricProgram( # minimize",
                          "          %s," % self.cost,
                          "          [   # subject to"] +
                         ["              %s," % constr
                          for constr in self.constraints] +
                         ['          ],',
                          "          substitutions={ %s }," %
                          pformat(self.substitutions, indent=26)[26:-1],
                          '          solver="%s")' % self.solver])

    def _latex(self, unused=None):
        """LaTeX representation of a GeometricProgram.
        Contains all of its parameters."""
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
                                 for var, val in self.substitutions.items()]) +
                         ["\\end{array}"])

    def test(self, solver=None, printing=False, skipfailures=False):
        end = " with default settings."
        try:
            self.solve(solver, printing, skipfailures)
            print(self.__class__.__name__, "solved successfully" + end)
        except Exception as e:
            print(self.__class__.__name__, "failed to solve" + end)
            self.checkbounds()
            raise(e)

    def solve(self, *args, **kwargs):
        return self._solve(*args, **kwargs)

    def _solve(self, solver=None, printing=True, skipfailures=False,
               allownonoptimal=False):
        """Solves a GeometricProgram and returns the solution.

        Arguments
        ---------
        printing : bool (optional)
            If True (default), then prints out solver used and time to solve.

        Returns
        -------
        solution : dict
            A dictionary containing the optimal values for each free variable.
        """
        if solver is None:
            solver = self.solver
        if solver == 'cvxopt':
            from ._cvxopt import cvxoptimize_fn
            solverfn = cvxoptimize_fn(self.options)
        elif solver == "mosek_cli":
            from ._mosek import cli_expopt
            filename = self.options.get('filename', 'gpkit_mosek')
            solverfn = cli_expopt.imize_fn(filename)
        elif solver == "mosek":
            from ._mosek import expopt
            solverfn = expopt.imize
        elif hasattr(solver, "__call__"):
            solverfn = solver
            solver = solver.__name__
        else:
            if not solver:
                raise ValueError("No solver was given; perhaps gpkit was not"
                                 " properly installed, or found no solvers"
                                 " during the install process.")
            raise ValueError("Solver %s is not implemented!" % solver)
        self.solverfn = solverfn
        self.solver = solver

        if printing:
            print("Using solver '%s'" % solver)
        self.starttime = time()

        if self.sweep:
            solution = self._solve_sweep(printing,
                                         skipfailures,
                                         allownonoptimal)
        else:
            if printing:
                print("Solving for %i variables." % len(self.varlocs))
            solution = GPSolutionArray()
            solution.gp = self
            solution.append(self._run_solver(allownonoptimal))
            solution.toarray()

        self.endtime = time()
        if printing:
            print("Solving took %.3g seconds."
                  % (self.endtime - self.starttime))
        self.solution = solution
        if hasattr(self, "calc"):
            self.result.update(self.calc(solution))
        # would love to return solution.update(result), but can't guarantee
        # that the result will be VarKey-based!
        return self.solution

    def _solve_sweep(self, printing, skipfailures, allownonoptimal):
        """Runs a GeometricProgram through a sweep, solving at each grid point

        Arguments
        ---------
        printing : bool (optional)
            If True, then prints out sweep and GeometricProgram size.

        Returns
        -------
        solution : dict
            A dictionary containing the array of optimal values
            for each free variable.
        """
        solution = GPSolutionArray()
        solution.gp = self

        self.presweep = self.last

        if len(self.sweep) == 1:
            sweep_grids = np.array(list(self.sweep.values()))
        else:
            sweep_grids = np.meshgrid(*list(self.sweep.values()))

        N_passes = sweep_grids[0].size
        sweep_vects = {var: grid.reshape(N_passes)
                       for (var, grid) in zip(self.sweep, sweep_grids)}
        if printing:
            print("Solving for %i variables over %i passes." % (
                  len(self.varlocs), N_passes))

        linkedsweep = self.linkedsweep

        def run_solver_i(i):
            this_pass = {var: sweep_vect[i]
                         for (var, sweep_vect) in sweep_vects.items()}
            linked = {var: fn(*[this_pass[VarKey(v)]
                                for v in var.descr["args"]])
                      for var, fn in linkedsweep.items()}
            this_pass.update(linked)
            self.sub(this_pass, frombase='presweep')
            if skipfailures:
                try:
                    return self._run_solver(allownonoptimal)
                except (RuntimeWarning, ValueError):
                    return None
            else:
                return self._run_solver(allownonoptimal)

        if POOL:
            mapfn = POOL.map_sync
        else:
            mapfn = map

        for sol in mapfn(run_solver_i, range(N_passes)):
            if sol is not None:
                solution.append(sol)

        solution.toarray()

        self.load(self.presweep)

        return solution

    def _run_solver(self, allownonoptimal):
        "Gets a solver's raw output, then checks and standardizes it."

        allpos = not any(mag(self.cs) <= 0)
        cs, exps, A, p_idxs, k, removed_idxs = self.genA(allpos=allpos)
        result = self.solverfn(c=cs, A=A, p_idxs=p_idxs, k=k)
        cs = np.array(cs)
        p_idxs = np.array(p_idxs)

        if allpos:
            unsubbedexps = self.unsubbed.exps
            unsubbedvarlocs = self.unsubbed.varlocs
        else:
            unsubbedexps = [exp for i, exp in enumerate(self.unsubbed.exps)
                            if i not in removed_idxs]
            unsubbedvarlocs, _ = locate_vars(unsubbedexps)

        return self._parse_result(result, unsubbedexps, unsubbedvarlocs,
                                  self.varlocs, cs, p_idxs, allownonoptimal)

    def _parse_result(self, result, unsubbedexps, unsubbedvarlocs, varlocs,
                      cs, p_idxs, allownonoptimal):
        if cs is None:
            cs = self.cs
        if p_idxs is None:
            p_idxs = self.p_idxs

        if result['status'] not in ["optimal", "OPTIMAL"]:
            if allownonoptimal:
                print("Nonoptimal result returned because 'allownonoptimal'"
                      " flag was set to True")
            else:
                raise RuntimeWarning("final status of solver '%s' was '%s', "
                                     "not 'optimal'." %
                                     (self.solver, result['status']) +
                                     "\n\nTo find a feasible solution to"
                                     " a relaxed version of your gp,"
                                     " run gpkit.find_feasible_point(gp).")

        variables = dict(zip(varlocs, np.exp(result['primal']).ravel()))
        variables.update(self.substitutions)

        # constraints must be within arbitrary epsilon 1e-4 of 1
        # takes a while to evaluate!
        # for p in self.constraints:
        #    val = p.subcmag(variables)
        #    if abs(val-1) > 1e-4:
        #        raise RuntimeWarning("constraint exceeded:"
        #                             " %s = 1 + %0.2e" % (p, val-1))

        if "objective" in result and "mosek" not in self.solver:
            # TODO remove mosek conditional -- issue # 296
            cost = float(result["objective"])
        else:
            cost = self.cost.subcmag(variables)

        sensitivities = {}
        if "nu" in result:
            nu = np.array(result["nu"]).ravel()
            la = np.array([sum(nu[p_idxs == i])
                           for i in range(len(self.posynomials))])
        elif "la" in result:
            la = np.array(result["la"]).ravel()
            if len(la) == len(self.posynomials) - 1:
                # assume the cost's sensitivity has been dropped
                la = np.hstack(([1.0], la))
            Ax = np.array(np.dot(self.A.todense(), result['primal'])).ravel()
            z = Ax + np.log(cs)
            m_iss = [p_idxs == i for i in range(len(la))]
            nu = np.hstack([la[p_i]*np.exp(z[m_is])/sum(np.exp(z[m_is]))
                            for p_i, m_is in enumerate(m_iss)])
        else:
            raise RuntimeWarning("the dual solution was not returned!")

        sensitivities["monomials"] = nu
        sensitivities["posynomials"] = la

        sens_vars = {var: (sum([unsubbedexps[i][var]*nu[i]
                                for i in locs]))
                     for (var, locs) in unsubbedvarlocs.items()}
        sens_vars.update({v: 0 for v in self.unsubbed.varlocs
                          if v not in sens_vars})
        sensitivities["variables"] = sens_vars

        # free-variable sensitivities must be < arbitrary epsilon 1e-4
        for var, S in sensitivities["variables"].items():
            if var not in self.substitutions and abs(S) > 1e-4:
                raise RuntimeWarning("free variable too sensitive:"
                                     " S_{%s} = %0.2e" % (var, S))

        local_exp = {var: S for (var, S) in sens_vars.items() if abs(S) >= 0.1}
        local_cs = (variables[var]**-S for (var, S) in local_exp.items())
        local_c = functools.reduce(mul, local_cs, cost)
        local_model = Monomial(local_exp, local_c)

        # vectorvar substitution
        for var in self.unsubbed.varlocs:
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

        constants = {var: val for var, val in variables.items()
                     if var in self.substitutions}
        free_variables = {var: val for var, val in variables.items()
                          if var not in self.substitutions}
        return dict(cost=cost,
                    variables=variables,
                    sensitivities=sensitivities,
                    local_model=local_model)
