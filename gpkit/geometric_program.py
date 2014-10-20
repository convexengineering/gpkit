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
import operator

from .models import Model
from .nomials import Constraint, MonoEQConstraint
from .nomials import Monomial

import gpkit.plotting


def flatten_constr(l):
    """Flattens an iterable that contains only constraints and other iterables

    Parameters
    ----------
    l : Iterable
        Top-level constraints container

    Returns
    -------
    out : list
        List of all constraints found in the nested iterables

    Raises
    ------
    TypeError
        If an object is found that is neither Constraint nor Iterable
    """
    out = []
    for el in l:
        if isinstance(el, Constraint):
            out.append(el)
        elif isinstance(el, Iterable):
            for elel in flatten_constr(el):
                out.append(elel)
        else:
            raise TypeError("The constraint list"
                            " %s contains invalid constraint '%s'." % (l, el))
    return out


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
        gp.solve()

    """

    def __init__(self, cost, constraints, substitutions={},
                 solver=None, options={}):
        self.cost = cost
        self.constraints = tuple(flatten_constr(constraints))
        posynomials = [self.cost]
        for constraint in self.constraints:
            if isinstance(constraint, MonoEQConstraint):
                posynomials += [constraint.leq, constraint.geq]
            else:
                posynomials.append(constraint)
        self.posynomials = tuple(posynomials)

        self.options = options
        if solver is not None:
            self.solver = solver
        else:
            from gpkit import settings
            self.solver = settings['installed_solvers'][0]

        self.vectorvars = {}
        self.sweep = {}
        self._gen_unsubbed_vars()

        if substitutions:
            self.sub(substitutions, tobase='initialsub')

    def __eq__(self, other):
        "GP equality is determined by their string representations."
        return str(self) == str(other)

    def __ne__(self, other):
        "GP inequality is determined by their string representations."
        return not self == other

    def __repr__(self):
        "The string representation of a GP contains all of its parameters."
        return "\n".join(["gpkit.GP( # minimize",
                          "          %s," % self.cost._string(),
                          "          [   # subject to"] +
                         ["              %s," % p._string()
                          for p in self.constraints] +
                         ['          ],',
                          "          substitutions={ %s }," %
                          pformat(self.substitutions, indent=26)[26:-1],
                          '          solver="%s")' % self.solver]
                         )

    def _latex(self, unused=None):
        "The LaTeX of a GP contains its cost and constraint posynomials"
        return "\n".join(["\\begin{array}[ll]",
                          "\\text{}",
                          "\\text{minimize}",
                          "    & %s \\\\" % self.cost._latex(),
                          "\\text{subject to}"] +
                         ["    & %s \\\\" % c._latex()
                          for c in self.constraints] +
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
        if printing: print("Using solver '%s'" % self.solver)
        self.starttime = time()

        if self.sweep:
            self.solution = self._solve_sweep(printing)
        else:
            self.solution = self.__run_solver()

        self.endtime = time()
        if printing:
            print("Solving took %.3g seconds   "
                  % (self.endtime - self.starttime))
        return self.solution

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

        sol_list = GPSolutionList()

        for i in range(N_passes):
            this_pass = {var: sweep_vect[i]
                         for (var, sweep_vect) in sweep_vects.items()}
            self.sub(this_pass, frombase='presweep', tobase='swept')
            sol = self.__run_solver()
            sol_list.append(sol)

        sol_list.toarray()

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
        """Checks and formats a solver's raw output.

        """
        # check solver status
        if result['status'] is not 'optimal':
            raise RuntimeWarning("final status of solver '%s' was '%s' not"
                                 "'optimal'." % (self.solver, result['status']))

        variables = dict(zip(self.var_locs, np.exp(result['primal'])))
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
        if "nu" not in result and "lambda" not in result:
            raise Exception("The dual solution was not returned!")
        if "nu" in result:
            sensitivities["monomials"] = np.array(result["nu"])
        else:
            pass  # generate nu from lambda
        if "la" in result:
            sensitivities["posynomials"] = np.array(result["la"])
        else:
            la = [sum(sensitivities["monomials"][np.array(self.p_idxs) == i])
                  for i in range(len(self.posynomials))]
            sensitivities["posynomials"] = np.array(la)

        sens_vars = {'%s' % var: (sum([self.unsubbed.exps[i][var]
                                       * sensitivities["monomials"][i]
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
        local_c = reduce(operator.mul, local_cs, cost)
        local_model = Monomial(local_exp, local_c)

        # vectorvar substitution
        for vectorvar, length in self.vectorvars.items():
            if vectorvar not in variables and length:
                vectorval, vectorS = [], []
                for i in range(length):
                    var = '{%s}_{%s}' % (vectorvar, i+1)
                    val = variables.pop(var)
                    S = sensitivities["variables"].pop(var)
                    vectorval.append(val)
                    vectorS.append(S)
                variables[vectorvar] = np.array(vectorval)
                sensitivities["variables"][vectorvar] = np.array(vectorS)

        return dict(cost=cost,
                    variables=variables,
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


class GPSolutionList(dict):
    # sol_array = GPSolutionArray(N_passes)
    # sol_array[i] = sol
    # sol_array.reshape(sweep_shape)

    def append(self, sol):
        if not hasattr(self, 'initialized'):
            enlist_dict(sol, self)
            self.initialized = True
        else:
            append_dict(sol, self)

    def get(self, i):
        return index_dict(i, self, {})

    def toarray(self, shape=None):
        if shape is None:
            enray_dict(self, self)


def enlist_dict(i, o):
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = enlist_dict(v, {})
        else:
            o[k] = [v]
    assert set(i.keys()) == set(o.keys())
    return o


def append_dict(i, o):
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = append_dict(v, o[k])
        else:
            o[k].append(v)
    assert set(i.keys()) == set(o.keys())
    return o


def index_dict(idx, i, o):
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = index_dict(idx, v, {})
        else:
            o[k] = v[idx]
    assert set(i.keys()) == set(o.keys())
    return o


def enray_dict(i, o):
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = enray_dict(v, {})
        else:
            o[k] = np.array(v)
    assert set(i.keys()) == set(o.keys())
    return o
