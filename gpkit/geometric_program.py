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
from .nomials import latex_num
from .nomials import Variable


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
    >>> gp.solve()

    """

    def __init__(self, cost, constraints, substitutions={},
                 solver=None, options={}):
        self.cost = cost
        # TODO: parse constraints during flattening, calling Posyarray on
        #       anything that holds only posys and then saving that list.
        #       This will allow prettier constraint printing.
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

        constants = {var: var.descr["value"] for var in self.var_locs
                     if "value" in var.descr}

        substitutions.update(constants)

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
            print("Solving took %.3g seconds   "
                  % (self.endtime - self.starttime))
        self.solution = GPSolutionArray(solution)
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
        sol_list = GPSolutionArray()

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
        if result['status'] not in ["optimal", "OPTIMAL"]:
            raise RuntimeWarning("final status of solver '%s' was '%s' not "
                                 "'optimal'" % (self.solver, result['status']))

        variables = dict(zip(self.var_locs,
                             np.exp(result['primal']).ravel()))
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
            if len(la) == len(self.posynomials) - 1 and la[0] != 1.0:
                la = np.hstack(([1.0], la))
            Ax = np.array(np.dot(self.A.todense(), result['primal'])).ravel()
            z = Ax + np.log(self.cs)
            mon_iss = [self.p_idxs == i for i in range(len(la))]
            nu = np.hstack([la[pos_i]*np.exp(z[mon_is])/sum(np.exp(z[mon_is]))
                            for pos_i, mon_is in enumerate(mon_iss)])
        else:
            raise Exception("The dual solution was not returned!")

        sensitivities["monomials"] = nu
        sensitivities["posynomials"] = la

        sens_vars = {var: (sum([self.unsubbed.exps[i][var]
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
        vardicts = [variables, sensitivities["variables"]]
        for var in self.unsubbed.var_locs:
            if "idx" in var.descr and "length" in var.descr:
                veckey = Variable(var.name, **var.descr)
                del veckey.descr["idx"]
                for vardict in vardicts:
                    if veckey not in vardict:
                        vardict[veckey] = np.empy(var.descr["length"]) + np.nan
                    vardict[veckey][var.descr["idx"]] = vardict.pop(var)

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


class DictOfLists(dict):
    "A hierarchy of dicionaries, with lists as the bottom."

    def append(self, sol):
        "Appends a dict (of dicts) of lists to all held lists."
        if not hasattr(self, 'initialized'):
            enlist_dict(sol, self)
            self.initialized = True
        else:
            append_dict(sol, self)

    def atindex(self, i):
        "Indexes into each list independently."
        return index_dict(i, self, {})

    def toarray(self, shape=None):
        "Converts all lists into arrays."
        if shape is None:
            enray_dict(self, self)


def enlist_dict(i, o):
    "Recursviely copies dict i into o, placing non-dict items into lists."
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = enlist_dict(v, {})
        else:
            o[k] = [v]
    assert set(i.keys()) == set(o.keys())
    return o


def append_dict(i, o):
    "Recursviely travels dict o and appends items found in i."
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = append_dict(v, o[k])
        else:
            o[k].append(v)
    assert set(i.keys()) == set(o.keys())
    return o


def index_dict(idx, i, o):
    "Recursviely travels dict i, placing items at idx into dict o."
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = index_dict(idx, v, {})
        else:
            o[k] = v[idx]
    assert set(i.keys()) == set(o.keys())
    return o


def enray_dict(i, o):
    "Recursively turns lists into numpy arrays."
    for k, v in i.items():
        if isinstance(v, dict):
            o[k] = enray_dict(v, {})
        else:
            o[k] = np.array(v)
    assert set(i.keys()) == set(o.keys())
    return o


class GPSolutionArray(DictOfLists):
    "DictofLists extended with posynomial substitution."

    def subinto(self, p):
        "Returns numpy array of each solution substituted into p."
        return np.array([p.sub(self.atindex(i)["variables"])
                         for i in range(len(self["cost"]))])

    def senssubinto(self, p):
        """Returns numpy array of each solution's sensitivity substituted into p.

        Each element must be scalar, so as to avoid any negative posynomials.
        """
        senssubbeds = [p.sub(self.atindex(i)["sensitivities"]["variables"],
                             allow_negative=True)
                       for i in range(len(self["cost"]))]
        if not all([isinstance(subbed, Monomial) for subbed in senssubbeds]):
            raise ValueError("senssub can only return scalars")
        if any([subbed.exp for subbed in senssubbeds]):
            raise ValueError("senssub can only return scalars")
        return np.array([subbed.c for subbed in senssubbeds], np.dtype('float'))

    def __str__(self):
        print_results_table(self["variables"], "Variable Value (average)")
        print_results_table(self["sensitivities"]["variables"],
                            "Variable Sensitivity (average)")


# there should be an interactive gpkit library
def print_results_table(data, title, senss=False):
    print("                    | " + title)
    for var, table in data.items():
        try:
            val = table.mean()
        except AttributeError:
            val = table
        if senss:
            units = "-"
            minval = 1e-2
        else:
            units = var.descr.get('units', '-')
            minval = None
        label = var.descr.get('label', '')
        if minval is None or abs(val) > minval:
            print("%19s" % var, ": %-8.3g" % val, "[%s] %s" % (units, label))
    print("                    |")
