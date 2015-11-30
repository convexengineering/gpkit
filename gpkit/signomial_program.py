"""Implement the SignomialProgram class"""
import numpy as np

from time import time
from functools import reduce as functools_reduce
from operator import mul

from .geometric_program import GeometricProgram
from .nomials import Signomial, PosynomialConstraint

from .feasibility import feasibility_model


class SignomialProgram(object):
    """Prepares a collection of signomials for a SP solve.

    Arguments
    ---------
    cost : Posynomial
        Objective to minimize when solving
    constraints : list of Constraint or SignomialConstraint objects
        Constraints to maintain when solving (implicitly Signomials <= 1)
    verbosity : int (optional)
        Currently has no effect: SignomialPrograms don't know anything new
        after being created, unlike GeometricPrograms.

    Attributes with side effects
    ----------------------------
    `gps` is set during a solve
    `result` is set at the end of a solve

    Examples
    --------
    >>> gp = gpkit.geometric_program.SignomialProgram(
                        # minimize
                        x,
                        [   # subject to
                            1/x - y/x,  # <= 1, implicitly
                            y/10  # <= 1
                        ])
    >>> gp.solve()
    """

    def __init__(self, cost, constraints, substitutions=None, verbosity=2):
        if cost.any_nonpositive_cs:
            raise TypeError("""SignomialPrograms need Posyomial objectives.

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable z to be greater than the desired Signomial objective s
    (z >= s) and then minimizing that dummy variable.""")

        self.cost = cost
        self.constraints = constraints
        self.posyconstraints = []
        self.localposyconstraints = []
        self.substitutions = substitutions if substitutions else {}

        for constraint in self.constraints:
            if substitutions:
                constraint.substitutions.update(substitutions)
            posy = False
            if hasattr(constraint, "as_localposyconstr"):
                posy = constraint.as_localposyconstr(None)
                if posy:
                    self.localposyconstraints.append(constraint)
            if not posy and hasattr(constraint, "as_posyslt1"):
                posy = constraint.as_posyslt1()
                if posy:
                    self.posyconstraints.append(constraint)
                else:
                    raise ValueError("%s is an invalid constraint for a"
                                     " SignomialProgram" % constraint)

        if not self.localposyconstraints:
            raise ValueError("SignomialPrograms must contain at least one"
                             " SignomialConstraint.")

    def localsolve(self, solver=None, verbosity=1, x0=None, rel_tol=1e-4,
                   iteration_limit=50, *args, **kwargs):
        """Locally solves a SignomialProgram and returns the solution.

        Arguments
        ---------
        solver : str or function (optional)
            By default uses one of the solvers found during installation.
            If set to "mosek", "mosek_cli", or "cvxopt", uses that solver.
            If set to a function, passes that function cs, A, p_idxs, and k.
        verbosity : int (optional)
            If greater than 0, prints solve time and number of iterations.
            Each GP is created and solved with verbosity one less than this, so
            if greater than 1, prints solver name and time for each GP.
        x0 : dict (optional)
            Initial location to approximate signomials about.
        rel_tol : float
            Iteration ends when this is greater than the distance between two
            consecutive solve's objective values.
        iteration_limit : int
            Maximum GP iterations allowed.
        *args, **kwargs :
            Passed to solver function.


        Returns
        -------
        result : dict
            A dictionary containing the translated solver result.
        """
        startpoint = x0 if x0 else {}
        if verbosity > 0:
            print("Beginning signomial solve.")
            self.starttime = time()
        self.gps = []  # NOTE: SIDE EFFECTS
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > rel_tol:
            if len(self.gps) > iteration_limit:
                raise RuntimeWarning("""problem unsolved after %s iterations.

    The last result is available in Model.program.gps[-1].result. If the gps
    appear to be converging, you may wish to increase the iteration limit by
    calling .localsolve(..., iteration_limit=NEWLIMIT).""" % len(self.gps))
            gp = self.step(x0, verbosity=verbosity-1)
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            try:
                result = gp.solve(solver, verbosity-1, *args, **kwargs)
            except (RuntimeWarning, ValueError):
                nearest_feasible = feasibility_model(gp, "max")
                self.gps.append(nearest_feasible)
                result = nearest_feasible.solve(verbosity=verbosity-1)
                result["cost"] = None
            x0 = result["variables"]
            prevcost, cost = cost, result["cost"]
            if prevcost and cost:
                rel_improvement = abs(prevcost-cost)/(prevcost + cost)
            else:
                rel_improvement = None
        # solved successfully!
        if verbosity > 0:
            print("Solving took %i GP solves" % len(self.gps)
                  + " and %.3g seconds." % (time() - self.starttime))

        constr_senss = result["sensitivities"]["constraints"]
        posyapproxs = gp.constraints[len(self.posyconstraints):]
        for i, posyapprox in enumerate(posyapproxs):
            constr = self.localposyconstraints[i]
            posyapprox_sens = constr_senss.pop(str(posyapprox))
            var_senss = result["sensitivities"]["constants"]
            constr_sens = constr.sp_sensitivities(posyapprox, posyapprox_sens,
                                                  var_senss)
            result["sensitivities"]["constraints"][str(constr)] = constr_sens

        result["signomialstart"] = startpoint
        self.result = result  # NOTE: SIDE EFFECTS
        return result

    def step(self, x0=None, verbosity=1):
        localposyconstraints = []
        for constraint in self.localposyconstraints:
            lpc = constraint.as_localposyconstr(x0)
            if not lpc:
                raise ValueError("%s is an invalid constraint for a"
                                 " SignomialProgram" % constraint)
            localposyconstraints.append(lpc)
        constraints = self.posyconstraints + localposyconstraints
        return GeometricProgram(self.cost, constraints, self.substitutions,
                                verbosity=verbosity)

    def __repr__(self):
        return "gpkit.%s(\n%s)" % (self.__class__.__name__, str(self))

    def __str__(self):
        """String representation of a SignomialProgram.

        Contains all of its parameters."""
        return "\n".join(["  # minimize",
                          "    %s," % self.cost,
                          "[ # subject to"] +
                         ["    %s," % constr
                          for constr in self.constraints] +
                         [']'])

    def latex(self):
        """LaTeX representation of a SignomialProgram.

        Contains all of its parameters."""
        posy_neg = []
        for p, n in zip(self.posynomials, self.negynomials[1:]):
            try:
                posy_neg.append((p.latex(), n.latex()))
            except:
                posy_neg.append((p.latex(), str(n)))
        return "\n".join(["\\begin{array}[ll]",
                          "\\text{}",
                          "\\text{minimize}",
                          "    & %s \\\\" % self.cost.latex(),
                          "\\text{subject to}"] +
                         ["    & %s \\\\" % constr
                          for constr in self.constraints] +
                         ["\\end{array}"])

    def _repr_latex_(self):
        return "$$"+self.latex()+"$$"
