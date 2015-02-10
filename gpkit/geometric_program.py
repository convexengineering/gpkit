# -*- coding: utf-8 -*-
"""Module for creating GP instances.

    Example
    -------
    >>> gp = gpkit.GP(cost, constraints, substitutions)

"""

import numpy as np
import itertools

from time import time
from pprint import pformat
from collections import Iterable, defaultdict
from functools import reduce
from operator import mul

from .small_classes import Strings
from .small_classes import DictOfLists
from .model import Model
from .nomials import Constraint, MonoEQConstraint
from .nomials import Monomial
from .nomials import VarKey
from .posyarray import PosyArray

from .small_scripts import latex_num
from .small_scripts import flatten
from .small_scripts import locate_vars
from .small_scripts import results_table
from .small_scripts import mag

try:
    from IPython.parallel import Client
    assert len(Client()) > 0
    pool = Client()[:]
    pool.use_dill()
    print "Using parallel execution of sweeps on %s clients" % len(Client())
except:
    pool = None


class GPSolutionArray(DictOfLists):
    "DictofLists extended with posynomial substitution."

    def __len__(self):
        try:
            return len(self["cost"])
        except TypeError:
            return 1

    def __call__(self, p):
        return self.subinto(p).c

    def getvars(self, *args):
        out = [self["variables"][arg] for arg in args]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def subinto(self, p):
        "Returns numpy array of each solution substituted into p."
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
                             allow_negative=True) for i in range(len(self))]
            assert all([isinstance(subbed, Monomial) for subbed in subbeds])
            assert not any([subbed.exp for subbed in subbeds])
            return np.array([mag(subbed.c) for subbed in subbeds],
                            np.dtype('float'))
        else:
            subbed = p.sub(self["sensitivities"]["variables"], allow_negative=True)
            assert isinstance(subbed, Monomial)
            assert not subbed.exp
            return subbed.c

    def table(self, tables=["cost", "free_variables",
                            "constants", "sensitivities"]):
        if isinstance(tables, Strings):
            tables = [tables]
        strs = []
        if "cost" in tables:
            if len(self) > 1:
                strs += ["%13.5g : Cost (average of %i values)" % (self["cost"].mean(), len(self))]
            else:
                strs += ["%13.5g : Cost" % self["cost"].mean()]
        if "free_variables" in tables:
            strs += [results_table(self["free_variables"],
                                   "Free variables" + ("" if len(self) == 1 else " (average)"))]
        if "constants" in tables:
            strs += [results_table(self["constants"], "Constants" + ("" if len(self) == 1 else " (average)"))]
        if "sensitivities" in tables:
            strs += [results_table(self["sensitivities"]["variables"],
                                   "Constant sensitivities" + ("" if len(self) == 1 else " (average)"),
                                   senss=True)]
        return "\n".join(strs)


class GP(Model):
    """Holds a model and cost function for passing to solvers.

    Parameters
    ----------
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
    >>> gp = gpkit.GP(  # minimize
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
            # setup cost/constraints replace init ones
            self.cost, constraints = self.setup(*args, **kwargs)
        else:
            if "cost" in kwargs:
                self.cost = kwargs["cost"]
            else:
                self.cost = args.pop(0)
            if "constraints" in kwargs:
                constraints = kwargs["constraints"]
            else:
                constraints = args.pop(0)
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
            modelname = self.__class__.__name__
            if modelname in GP.model_nums:
                modelname += str(GP.model_nums[modelname])
            GP.model_nums[modelname] += 1
            for p in self.posynomials:
                for k in p.var_locs:
                    if not "model" in k.descr:
                        newk = VarKey(k, model=modelname)
                        p.var_locs[newk] = p.var_locs.pop(k)
                for exp in p.exps:
                    for k in exp:
                        if not "model" in k.descr:
                            newk = VarKey(k, model=modelname)
                            exp[newk] = exp.pop(k)

        self.sweep = {}
        self._gen_unsubbed_vars(printing=False)
        values = {var: var.descr["value"]
                  for var in self.var_locs if "value" in var.descr}
        if "substitutions" in kwargs:
            values.update(kwargs["substitutions"])
        elif len(args) > 0 and not hasattr(self, "setup"):
            values.update(args.pop(0))
        if values:
            self.sub(values)
            self.initalsub = self.last
        else:
            self._gen_unsubbed_vars(printing=True)

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
        if isinstance(other, GP):
            # don't add costs b/c that breaks when costs have units
            return GP(Monomial(1), self.constraints + other.constraints)
        else:
            return NotImplemented

    def __call__(self, key):
        return self[key]

    def vars(self, *args):
        out = [self[arg] for arg in args]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def __getitem__(self, key):
        for attr in ["result", "solution", "solv", "variables"]:
            if hasattr(self, attr) or (attr == "solv" and hasattr(self, "solution")):
                if attr == "solv":
                    d = self.solution["variables"]
                else:
                    d = getattr(self, attr)
                if key in d:
                    return d[key]
                elif key+"_0" in d:
                    maybevec = self.varkeys[key+"_0"].descr
                    if "length" in maybevec:
                        # then it is a vector!
                        length = maybevec["length"]
                        l = [d[key+"_%s" % i] for i in range(length)]
                        if attr == "variables":
                            return PosyArray(l)
                        else:
                            return np.array(l)


        raise KeyError("'%s' was not found as a result solution or variable."  % key)

    def __setitem__(self, key, value):
        self.results[key] = value

    def __eq__(self, other):
        "GP equality is determined by their string representations."
        return str(self) == str(other)

    def __ne__(self, other):
        "GP inequality is determined by their string representations."
        return not self == other

    def __repr__(self):
        "The string representation of a GP contains all of its parameters."
        return "\n".join(["gpkit.GP( # minimize",
                          "          %s," % self.cost,
                          "          [   # subject to"] +
                         ["              %s," % constr
                          for constr in self.constraints] +
                         ['          ],',
                          "          substitutions={ %s }," %
                          pformat(self.substitutions, indent=26)[26:-1],
                          '          solver="%s")' % self.solver]
                         )

    def _latex(self, unused=None):
        "The LaTeX representation of a GP contains all of its parameters."
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
        try:
            self.solve(solver, printing, skipfailures)
            print self.__class__.__name__, "solved successfully with default settings."
        except Exception, e:
            print self.__class__.__name__, "failed to solve with default settings."
            self.checkbounds()
            raise(e)

    def solve(self, solver=None, printing=True, skipfailures=False):
        """Solves a GP and returns the solution.

        Parameters
        ----------
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
            solverfn = cvxoptimize_fn(self.k, self.options)
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
            raise ValueError("Solver %s is not implemented!" % solver)
        self.solverfn = solverfn
        self.solver = solver

        if printing:
            print("Using solver '%s'" % solver)
        self.starttime = time()

        if self.sweep:
            solution = self._solve_sweep(printing, skipfailures)
        else:
            solution = GPSolutionArray()
            solution.append(self.__run_solver())
            solution.toarray()

        self.endtime = time()
        if printing:
            print("Solving took %.3g seconds"
                  % (self.endtime - self.starttime))
        self.solution = solution
        if hasattr(self, "calc"):
            self.result = self.calc(solution)
        # would love to return solution.update(result), but can't guarantee
        # that the result will be VarKey-based!
        return self.solution

    def _solve_sweep(self, printing, skipfailures):
        """Runs a GP through a sweep, solving at each grid point

        Parameters
        ----------
        printing : bool (optional)
            If True, then prints out sweep and GP size.

        Returns
        -------
        solution : dict
            A dictionary containing the array of optimal values
            for each free variable.
        """
        solution = GPSolutionArray()

        self.presweep = self.last
        self.sub({var: 1.0 for var in self.sweep})

        if len(self.sweep) == 1:
            sweep_grids = np.array(self.sweep.values())
        else:
            sweep_grids = np.meshgrid(*self.sweep.values())
        N_passes = sweep_grids[0].size
        sweep_vects = {var: grid.reshape(N_passes)
                       for (var, grid) in zip(self.sweep, sweep_grids)}
        if printing:
            print("Sweeping %i variables over %i passes" % (
                  len(self.sweep), N_passes))

        def run_solver_i(i):
            this_pass = {var: sweep_vect[i]
                         for (var, sweep_vect) in sweep_vects.items()}
            self.sub(this_pass, frombase='presweep')
            if skipfailures:
                try:
                    return self.__run_solver()
                except RuntimeWarning:
                    return None
            else:
                return self.__run_solver()

        if pool:
            mapfn = pool.map_sync
        else:
            mapfn = map

        for sol in mapfn(run_solver_i,  range(N_passes)):
            if sol is not None:
                solution.append(sol)

        solution.toarray()

        self.load(self.presweep)

        return solution

    def __run_solver(self):
        "Gets a solver's raw output, then checks and standardizes it."

        result = self.solverfn(self.cs, self.A, self.p_idxs)
        if result['status'] not in ["optimal", "OPTIMAL"]:
            raise RuntimeWarning("final status of solver '%s' was '%s' not "
                                 "'optimal'" % (self.solver, result['status']))

        variables = dict(zip(self.var_locs, np.exp(result['primal']).ravel()))
        variables.update(self.substitutions)

        # constraints must be within arbitrary epsilon 1e-4 of 1
        # takes a while to evaluate!
        # for p in self.constraints:
        #    val = p.subcmag(variables)
        #    if abs(val-1) > 1e-4:
        #        raise RuntimeWarning("constraint exceeded:"
        #                             " %s = 1 + %0.2e" % (p, val-1))

        if "objective" in result:
            cost = float(result["objective"])
        else:
            cost = self.cost.subcmag(variables)

        sensitivities = {}
        if "nu" in result:
            nu = np.array(result["nu"]).ravel()
            la = np.array([sum(nu[self.p_idxs == i])
                           for i in range(len(self.posynomials))])
        elif "la" in result:
            la = np.array(result["la"]).ravel()
            if len(la) == len(self.posynomials) - 1:
                # assume the cost's sensitivity has been dropped
                la = np.hstack(([1.0], la))
            Ax = np.array(np.dot(self.A.todense(), result['primal'])).ravel()
            z = Ax + np.log(self.cs)
            m_iss = [self.p_idxs == i for i in range(len(la))]
            nu = np.hstack([la[p_i]*np.exp(z[m_is])/sum(np.exp(z[m_is]))
                            for p_i, m_is in enumerate(m_iss)])
        else:
            raise RuntimeWarning("the dual solution was not returned!")

        sensitivities["monomials"] = nu
        sensitivities["posynomials"] = la

        sens_vars = {var: (sum([self.unsubbed.exps[i][var]*nu[i]
                                for i in locs]))
                     for (var, locs) in self.unsubbed.var_locs.items()}
        sensitivities["variables"] = sens_vars

        # free-variable sensitivities must be < arbitrary epsilon 1e-4
        for var, S in sensitivities["variables"].items():
            if var not in self.substitutions and abs(S) > 1e-4:
                raise RuntimeWarning("free variable too sensitive:"
                                     " S_{%s} = %0.2e" % (var, S))

        local_exp = {var: S for (var, S) in sens_vars.items() if abs(S) >= 0.1}
        local_cs = (variables[var]**-S for (var, S) in local_exp.items())
        local_c = reduce(mul, local_cs, cost)
        local_model = Monomial(local_exp, local_c)

        # vectorvar substitution
        for var in self.unsubbed.var_locs:
            if "idx" in var.descr and "length" in var.descr:
                descr = dict(var.descr)
                descr.pop("idx")
                units = descr.pop("units")
                veckey = VarKey(**descr)
                veckey.descr["units"] = units

                if veckey not in variables:
                    variables[veckey] = np.empty(var.descr["length"]) + np.nan
                variables[veckey][var.descr["idx"]] = variables.pop(var)

                if veckey not in sensitivities["variables"]:
                    sensitivities["variables"][veckey] = \
                        np.empty(var.descr["length"]) + np.nan
                sensitivities["variables"][veckey][var.descr["idx"]] = \
                    sensitivities["variables"].pop(var)

        constants = {var: val for var, val in variables.items()
                     if var in self.substitutions}
        free_variables = {var: val for var, val in variables.items()
                          if var not in self.substitutions}

        return dict(cost=cost,
                    variables=variables,
                    free_variables=free_variables,
                    constants=constants,
                    sensitivities=sensitivities,
                    local_model=local_model)
