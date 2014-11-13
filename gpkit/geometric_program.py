# -*- coding: utf-8 -*-
"""Module for creating GP instances.

    Example
    -------
    >>> gp = gpkit.GP(cost, constraints, substitutions)

"""

import numpy as np

from time import time
from pprint import pformat
from collections import Iterable
from functools import reduce
from operator import mul

from .small_classes import Strings
from .small_classes import DictOfLists
from .model import Model
from .nomials import Constraint, MonoEQConstraint
from .nomials import Monomial
from .nomials import Variable

from .small_scripts import latex_num
from .small_scripts import flatten
from .small_scripts import locate_vars
from .small_scripts import print_results_table


class GPSolutionArray(DictOfLists):
    "DictofLists extended with posynomial substitution."

    def subinto(self, p):
        "Returns numpy array of each solution substituted into p."
        return np.array([p.sub(self.atindex(i)["variables"])
                         for i in range(len(self["cost"]))])

    def senssubinto(self, p):
        """Returns array of each solution's sensitivity substituted into p

        Returns only scalar values.
        """
        subbeds = [p.sub(self.atindex(i)["sensitivities"]["variables"],
                         allow_negative=True) for i in range(len(self))]
        assert all([isinstance(subbed, Monomial) for subbed in subbeds])
        assert not any([subbed.exp for subbed in subbeds])
        return np.array([subbed.c for subbed in subbeds], np.dtype('float'))

    def print_table(self, tables=["cost", "free_variables",
                                  "constants", "sensitivities"]):
        if isinstance(tables, Strings):
            tables = [tables]
        if "cost" in tables:
            print("         %10.5g : Cost (mean)" % self["cost"].mean())
        if "free_variables" in tables:
            print_results_table(self["free_variables"], "Solution (mean)")
        if "constants" in tables:
            print_results_table(self["constants"], "Constants (mean)")
        if "sensitivities" in tables:
            print_results_table(self["sensitivities"]["variables"],
                                "Constants Sensitivities (mean)", senss=True)


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

    def __init__(self, cost, constraints, substitutions={},
                 solver=None, options={}):
        self.cost = cost
        # TODO: parse constraints during flattening, calling Posyarray on
        #       anything that holds only posys and then saving that list.
        #       This will allow prettier constraint printing.
        self.constraints = tuple(flatten(constraints, Constraint))
        posynomials = [self.cost]
        for constraint in self.constraints:
            if isinstance(constraint, MonoEQConstraint):
                posynomials += [constraint.leq, constraint.geq]
            else:
                posynomials.append(constraint)
        self.posynomials = tuple(posynomials)

        self.sweep = {}
        self._gen_unsubbed_vars()
        substitutions.update({var: var.descr["value"] for var in self.var_locs
                             if "value" in var.descr})
        if substitutions:
            self.sub(substitutions, tobase='initialsub')

        if solver is None:
            from gpkit import settings
            solver = settings['installed_solvers'][0]
        self.solver = solver
        self.options = options

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
                         sorted(["    & %s = %s \\\\" % (var._latex(), latex_num(val))
                                 for var, val in self.substitutions.items()]) +
                         ["\\end{array}"])

    def solve(self, printing=True):
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
        if printing:
            print("Using solver '%s'" % self.solver)
        self.starttime = time()

        if self.sweep:
            solution = self._solve_sweep(printing)
        else:
            solution = GPSolutionArray()
            solution.append(self.__run_solver())
            solution.toarray()

        self.endtime = time()
        if printing:
            print("Solving took %.3g seconds"
                  % (self.endtime - self.starttime))
        self.solution = solution
        return solution

    def _solve_sweep(self, printing):
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
        self.sub({var: 1 for var in self.sweep}, tobase='swept')

        sweep_dims = len(self.sweep)
        if sweep_dims == 1:
            sweep_grids = np.array(self.sweep.values())
        else:
            sweep_grids = np.meshgrid(*self.sweep.values())
        sweep_shape = sweep_grids[0].shape
        N_passes = sweep_grids[0].size
        if printing:
            print("Sweeping %i variables over %i passes" % (
                  sweep_dims, N_passes))
        sweep_grids = dict(zip(self.sweep, sweep_grids))
        sweep_vects = {var: grid.reshape(N_passes)
                       for (var, grid) in sweep_grids.items()}

        for i in range(N_passes):
            this_pass = {var: sweep_vect[i]
                         for (var, sweep_vect) in sweep_vects.items()}
            self.sub(this_pass, frombase='presweep', tobase='swept')
            sol = self.__run_solver()
            solution.append(sol)

        solution.toarray()

        self.load(self.presweep)

        return sol_list

    def __run_solver(self):
        "Switches between solver options"

        if self.solver == 'cvxopt':
            result = cvxoptimize(self.cs,
                                 self.A,
                                 self.k,
                                 self.options)
        elif self.solver == "mosek_cli":
            from ._mosek import cli_expopt
            filename = self.options.get('filename', 'gpkit_mosek')
            result = cli_expopt.imize(self.cs,
                                      self.A,
                                      self.p_idxs,
                                      filename)
        elif self.solver == "mosek":
            from ._mosek import expopt
            result = expopt.imize(self.cs,
                                  self.A,
                                  self.p_idxs)
        elif self.solver == "attached":
            result = self.options['solver'](self.cs,
                                            self.A,
                                            self.p_idxs,
                                            self.k)
        else:
            raise Exception("Solver %s is not implemented!" % self.solver)

        return self.__parse_result(result)

    def __parse_result(self, result):
        "Checks and formats a solver's raw output."

        if result['status'] not in ["optimal", "OPTIMAL"]:
            raise RuntimeWarning("final status of solver '%s' was '%s' not "
                                 "'optimal'" % (self.solver, result['status']))

        variables = dict(zip(self.var_locs, np.exp(result['primal']).ravel()))
        variables.update(self.substitutions)

        # constraints must be within arbitrary epsilon 1e-4 of 1
        for p in self.constraints:
            val = p.sub(variables).c
            if not val <= 1 + 1e-4:
                raise RuntimeWarning("constraint exceeded:"
                                     " %s = 1 + %0.2e" % (p, val-1))

        if "objective" in result:
            cost = float(result["objective"])
        else:
            costm = self.cost.sub(variables)
            assert costm.exp == {}
            cost = costm.c

        sensitivities = {}
        if "nu" in result:
            nu = np.array(result["nu"]).ravel()
            la = np.array([sum(nu[self.p_idxs == i])
                           for i in range(len(self.posynomials))])
        elif "la" in result:
            la = np.array(result["la"]).ravel()
            # check if cost's sensitivity has been dropped
            if len(la) == len(self.posynomials) - 1 and la[0] != 1.0:
                la = np.hstack(([1.0], la))
            Ax = np.array(np.dot(self.A.todense(), result['primal'])).ravel()
            z = Ax + np.log(self.cs)
            m_iss = [self.p_idxs == i for i in range(len(la))]
            nu = np.hstack([la[p_i]*np.exp(z[m_is])/sum(np.exp(z[m_is]))
                            for p_i, m_is in enumerate(m_iss)])
        else:
            raise Exception("The dual solution was not returned!")

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
        vardicts = [variables, sensitivities["variables"]]
        for var in self.unsubbed.var_locs:
            if "idx" in var.descr and "length" in var.descr:
                veckey = Variable(var.name, **var.descr)
                del veckey.descr["idx"]
                for vardict in vardicts:
                    if veckey not in vardict:
                        vardict[veckey] = np.empy(var.descr["length"]) + np.nan
                    vardict[veckey][var.descr["idx"]] = vardict.pop(var)

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


def cvxoptimize(c, A, k, options):
    """Interface to the CVXOPT solver

        Definitions
        -----------
        "[a,b] array of floats" indicates array-like data with shape [a,b]
        n is the number of monomials in the gp
        m is the number of variables in the gp
        p is the number of posynomials in the gp

        Parameters
        ----------
        c : floats array of shape n
            Coefficients of each monomial
        A: floats array of shape (m,n)
            Exponents of the various free variables for each monomial.
        p_idxs: ints array of shape n
            Posynomial index of each monomial

        Returns
        -------
        dict
            Contains the following keys
                "success": bool
                "objective_sol" float
                    Optimal value of the objective
                "primal_sol": floats array of size m
                    Optimal value of the free variables. Note: not in logspace.
                "dual_sol": floats array of size p
                    Optimal value of the dual variables, in logspace.
    """
    from cvxopt import solvers, spmatrix, matrix, log, exp
    solvers.options.update({'show_progress': False})
    solvers.options.update(options)
    g = log(matrix(c))
    F = spmatrix(A.data, A.row, A.col, tc='d')
    solution = solvers.gp(k, F, g)
    return dict(status=solution['status'],
                primal=solution['x'],
                la=solution['znl'])
