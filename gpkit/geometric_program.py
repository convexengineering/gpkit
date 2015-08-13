"""Implement the GeometricProgram class"""
import numpy as np

from time import time
from functools import reduce as functools_reduce
from operator import add

from .variables import Variable, VectorVariable
from .small_classes import CootMatrix
from .small_scripts import locate_vars
from .small_scripts import mag


class GeometricProgram(object):
    """Standard mathematical representation of a GP.

    Arguments
    ---------
    cost : Constraint
        Posynomial to minimize when solving
    constraints : list of Posynomials
        Constraints to maintain when solving (implicitly Posynomials <= 1)
        GeometricProgram does not accept equality constraints (e.g. x == 1);
         instead use two inequality constraints (e.g. x <= 1, 1/x <= 1)
    verbosity : int (optional)
        If verbosity is greater than zero, warns about missing bounds
        on creation.

    Attributes with side effects
    ----------------------------
    `solver_out` is set during a solve
    `solver_log` is set during a solve
    `result` is set at the end of a solve

    Examples
    --------
    >>> gp = gpkit.geometric_program.GeometricProgram(
                        # minimize
                        x,
                        [   # subject to
                            1/x  # <= 1, implicitly
                        ])
    >>> gp.solve()
    """

    def __init__(self, cost, constraints, verbosity=1):
        self.cost = cost
        self.constraints = constraints
        self.posynomials = [cost] + list(constraints)
        self.cs = np.hstack((mag(p.cs) for p in self.posynomials))
        if not all(self.cs > 0):
            raise ValueError("GeometricPrograms cannot contain Signomials.")
        self.exps = functools_reduce(add, (p.exps for p in self.posynomials))
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

        self.A, self.missingbounds = genA(self.exps, self.varlocs)

        if verbosity > 0:
            for var, bound in sorted(self.missingbounds.items()):
                print("%s has no %s bound" % (var, bound))

    def solve(self, solver=None, verbosity=1, *args, **kwargs):
        """Solves a GeometricProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses one of the solvers found during installation.
            If set to "mosek", "mosek_cli", or "cvxopt", uses that solver.
            If set to a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (optional)
            If greater than 0, prints solver name and solve time.
        *args, **kwargs :
            Passed to solver constructor and solver function.


        Returns
        -------
        result : dict
            A dictionary containing the translated solver result; keys below.

            cost : float
                The value of the objective at the solution.
            variables : dict
                The value of each variable at the solution.
            sensitivities : dict
                monomials : array of floats
                    Each monomial's dual variable value at the solution.
                posynomials : array of floats
                    Each posynomials's dual variable value at the solution.
        """
        if solver is None:
            from . import settings
            if settings['installed_solvers']:
                solver = settings['installed_solvers'][0]
            else:
                raise ValueError("No solver was given; perhaps gpkit was not"
                                 " properly installed, or found no solvers"
                                 " during the installation process.")

        if solver == 'cvxopt':
            from ._cvxopt import cvxoptimize_fn
            solverfn = cvxoptimize_fn(*args, **kwargs)
        elif solver == "mosek_cli":
            from ._mosek import cli_expopt
            solverfn = cli_expopt.imize_fn(*args, **kwargs)
        elif solver == "mosek":
            from ._mosek import expopt
            solverfn = expopt.imize
        elif hasattr(solver, "__call__"):
            solverfn = solver
            solver = solver.__name__
        else:
            raise ValueError("Unknown solver '%s'." % solver)

        if verbosity > 0:
            print("Using solver '%s'" % solver)
            self.starttime = time()
            print("Solving for %i variables." % len(self.varlocs))

        solver_out = solverfn(c=self.cs, A=self.A, p_idxs=self.p_idxs,
                              k=self.k, *args, **kwargs)

        self.solver_out = solver_out  # NOTE: SIDE EFFECTS
        # TODO: add a solver_log file using stream debugger

        if verbosity > 0:
            print("Solving took %.3g seconds." % (time() - self.starttime))

        result = {}
        # confirm lengths before calling zip
        assert len(self.varlocs) == len(solver_out['primal'])
        result["variables"] = dict(zip(self.varlocs,
                                       np.exp(solver_out['primal']).ravel()))
        if "objective" in solver_out:
            result["cost"] = float(solver_out["objective"])
        else:
            result["cost"] = self.cost.subcmag(result["variables"])

        result["sensitivities"] = {}
        if "nu" in solver_out:
            nu = np.ravel(solver_out["nu"])
            la = np.array([sum(nu[self.p_idxs == i])
                           for i in range(len(self.posynomials))])
        elif "la" in solver_out:
            la = np.ravel(solver_out["la"])
            if len(la) == len(self.posynomials) - 1:
                # assume the cost's sensitivity has been dropped
                la = np.hstack(([1.0], la))
            Ax = np.ravel(self.A.todense().dot(solver_out['primal']))
            z = Ax + np.log(self.cs)
            m_iss = [self.p_idxs == i for i in range(len(la))]
            nu = np.hstack([la[p_i]*np.exp(z[m_is])/sum(np.exp(z[m_is]))
                            for p_i, m_is in enumerate(m_iss)])
        else:
            raise RuntimeWarning("The dual solution was not returned.")
        result["sensitivities"]["monomials"] = nu
        result["sensitivities"]["posynomials"] = la

        self.result = result  # NOTE: SIDE EFFECTS

        if solver_out.get("status", None) not in ["optimal", "OPTIMAL"]:
            raise RuntimeWarning("final status of solver '%s' was '%s', "
                                 "not 'optimal'." %
                                 (solver, solver_out.get("status", None)) +
                                 "\n\nTo generate a feasibility-finding"
                                 " relaxed version of your\nGeometric Program,"
                                 " run model.program.feasibility_search()."
                                 "\n\nThe infeasible solve's result is stored"
                                 " in the 'result' attribute\n"
                                 "(model.program.result)"
                                 " and its raw output in 'solver_out'.")
        else:
            return result

    def feasibility_search(self, flavour="max", varname=None, *args, **kwargs):
        """Returns a new GP for the closest feasible point of the current GP.

        Arguments
        ---------
        flavour : str
            Specifies the objective function minimized in the search:

            "max" (default) : Apply the same slack to all constraints and
                              minimize that slack. Described in Eqn. 10
                              of [Boyd2007].

            "product" : Apply a unique slack to all constraints and minimize
                        the product of those slacks. Useful for identifying the
                        most problematic constraints. Described in Eqn. 11
                        of [Boyd2007]

        varname : str
            LaTeX name of slack variables.

        *args, **kwargs
            Passed on to GP initialization.

        [Boyd2007] : "A tutorial on geometric programming", Optim Eng 8:67-122

        """

        if flavour == "max":
            slackvar = Variable(varname)
            gp = GeometricProgram(slackvar,
                                  [1/slackvar] +  # slackvar > 1
                                  [constraint/slackvar # constraint <= sv
                                   for constraint in self.constraints],
                                  *args, **kwargs)
        elif flavour == "product":
            slackvars = VectorVariable(len(self.constraints), varname)
            gp = GeometricProgram(np.prod(slackvars),
                                  (1/slackvars).tolist() +  # slackvars > 1
                                  [constraint/slackvars[i]  # constraint <= sv
                                   for i, constraint in
                                   enumerate(self.constraints)],
                                  *args, **kwargs)
        else:
            raise ValueError("'%s' is an unknown flavour of feasibility." %
                             flavour)
        return gp

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
    """Generates A matrix from exps and varlocs

    Arguments
    ---------
        exps : list of Hashvectors
            Exponents for each monomial in a GP
        varlocs : dict
            Locations of each variable in exps

    Returns
    -------
        A : sparse Cootmatrix
            Exponents of the various free variables for each monomial: rows
            of A are monomials, columns of A are variables.
        missingbounds : dict
            Keys: variables that lack bounds. Values: which bounds are missed.
    """

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
            else:
                # just being safe
                raise RuntimeWarning("Unexpected varsign %s" % varsign)
            missingbounds[var] = bound

    # add constant terms
    for i, exp in enumerate(exps):
        if not exp:
            A.append(i, 0, 0)

    A.update_shape()

    return A, missingbounds
