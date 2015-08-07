import numpy as np

from time import time
from functools import reduce as functools_reduce
from operator import add

from .small_classes import CootMatrix
from .small_scripts import locate_vars
from .small_scripts import mag


class GeometricProgram(object):

    def __init__(self, cost, constraints, printing=True):
        self.cost = cost
        self.constraints = constraints
        self.posynomials = [cost] + list(constraints)
        self.cs = np.hstack((mag(p.cs) for p in self.posynomials))
        if not all([c >= 0 for c in self.cs]):
            raise ValueError("GeometricPrograms cannot contain coefficients"
                             "less than or equal to zero.")
        self.exps = functools_reduce(add, (x.exps for x in self.posynomials))
        self.varlocs, self.varkeys = locate_vars(self.exps)
        # k [j]: number of monomials (columns of F) present in each constraint
        self.k = [len(p.cs) for p in self.posynomials]
        # p_idxs [i]: posynomial index of each monomial
        p_idx = 0
        p_idxs = []
        for p_len in self.k:
            p_idxs += [p_idx]*p_len
            p_idx += 1
        self.p_idxs = np.array(p_idxs)

        self.A, missingbounds = genA(self.exps, self.varlocs)

        if printing:
            for var, bound in sorted(missingbounds.items()):
                print("%s has no %s bound" % (var, bound))

    def solve(self, solver=None, printing=True, skipfailures=False,
              options={}, *args, **kwargs):
        """Solves a GeometricProgram and returns the solution.

        Arguments
        ---------
        printing : bool (optional)
            If True (default), then prints out solver used and time to solve.

        Returns
        -------
        result : dict
            A dictionary containing the translated solver result.
        """
        if solver is None:
            from . import settings
            solver = settings['installed_solvers'][0]
        if solver == 'cvxopt':
            from ._cvxopt import cvxoptimize_fn
            solverfn = cvxoptimize_fn(options)
        elif solver == "mosek_cli":
            from ._mosek import cli_expopt
            filename = options.get('filename', 'gpkit_mosek')
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

        if printing:
            print("Using solver '%s'" % solver)
            self.starttime = time()
            print("Solving for %i variables." % len(self.varlocs))

        solver_out = solverfn(self.cs, self.A, self.p_idxs, self.k)
        # TODO: add 'options' argument for coordinating messaging etc

        if printing:
            print("Solving took %.3g seconds." % (time() - self.starttime))

        result = {}
        result["variables"] = dict(zip(self.varlocs,
                                   np.exp(solver_out['primal']).ravel()))
        if "objective" in solver_out:
            result["cost"] = float(solver_out["objective"])
        else:
            result["cost"] = self.cost.subcmag(result["variables"])

        result["sensitivities"] = {}
        if "nu" in solver_out:
            nu = np.array(solver_out["nu"]).ravel()
            la = np.array([sum(nu[self.p_idxs == i])
                           for i in range(len(self.posynomials))])
        elif "la" in solver_out:
            la = np.array(solver_out["la"]).ravel()
            if len(la) == len(self.posynomials) - 1:
                # assume the cost's sensitivity has been dropped
                la = np.hstack(([1.0], la))
            Ax = np.array(self.A.todense().dot(solver_out['primal'])).ravel()
            z = Ax + np.log(self.cs)
            m_iss = [self.p_idxs == i for i in range(len(la))]
            nu = np.hstack([la[p_i]*np.exp(z[m_is])/sum(np.exp(z[m_is]))
                            for p_i, m_is in enumerate(m_iss)])
        else:
            raise RuntimeWarning("the dual solution was not returned!")
        result["sensitivities"]["monomials"] = nu
        result["sensitivities"]["posynomials"] = la

        # SIDE EFFECTS AHOY
        self.result = result
        self.solver_out = solver_out
        # TODO: add a solver_log file using stream debugger

        if solver_out.get("status", None) not in ["optimal", "OPTIMAL"]:
            raise RuntimeWarning("final status of solver '%s' was '%s', "
                                 "not 'optimal'." %
                                 (solver, solver_out.get("status", None)) +
                                 "\n\nTo find a feasible solution to a"
                                 " relaxed version of your Geometric Program,"
                                 "\nrun gpkit.find_feasible_point(model.program)."
                                 "\n\nThe infeasible solve's result is stored"
                                 "in the 'result' attribute (model.program.result)"
                                 "\nand its raw output in 'solver_out'.")
        else:
            return result

    def __repr__(self):
        return "gpkit.%s(\n%s)" % (self.__class__.__name__, str(self))

    def __str__(self):
        """String representation of a GeometricProgram.

        Contains all of its parameters."""
        return "\n".join(["  # minimize",
                          "    %s," % self.cost,
                          "[ # subject to"] +
                         ["    %s <= 1," % constr
                          for constr in self.constraints] +
                         [']'])

    def _latex(self, unused=None):
        """LaTeX representation of a GeometricProgram.

        Contains all of its parameters."""
        return "\n".join(["\\begin{array}[ll]",
                          "\\text{}",
                          "\\text{minimize}",
                          "    & %s \\\\" % self.cost._latex(),
                          "\\text{subject to}"] +
                         ["    & %s \\leq 1\\\\" % constr._latex()
                          for constr in self.constraints] +
                         ["\\end{array}"])


def genA(exps, varlocs):
    # A: exponents of the various free variables for each monomial
    #    rows of A are variables, columns are monomials

    missingbounds = {}
    A = CootMatrix([], [], [])
    for j, var in enumerate(varlocs):
        varsign = "both" if "value" in var.descr else None
        for i in varlocs[var]:
            exp = exps[i][var]
            A.append(i, j, exp)
            if varsign is "both":
                pass
            elif varsign is None:
                varsign = np.sign(exp)
            elif np.sign(exp) != varsign:
                varsign = "both"

        if varsign != "both":
            if varsign == 1:
                bound = "lower"
            elif varsign == -1:
                bound = "upper"
            missingbounds[var] = bound

    # add subbed-out monomials at the end
    if not exps[-1]:
        A.append(0, len(exps)-1, 0)

    A.update_shape()

    return A, missingbounds
