"""Implement the SignomialProgram class"""
from time import time
from ..geometric_program import GeometricProgram
from ..feasibility import feasibility_model
from .costed import CostedConstraintSet


class SignomialProgram(CostedConstraintSet):
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

    def __init__(self, cost, constraints, substitutions=None, verbosity=1):
        # pylint: disable=unused-argument
        "Constructor. Required for objects inheriting from np.ndarray."
        if cost.any_nonpositive_cs:
            raise TypeError("""SignomialPrograms need Posyomial objectives.

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable z to be greater than the desired Signomial objective s
    (z >= s) and then minimizing that dummy variable.""")
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        try:
            _ = self.as_posyslt1()  # should raise an error
            # TODO: is there a faster way to check?
        except TypeError:
            pass
        else:  # this is a GP
            raise ValueError("""No Signomials remained after substitution.

    SignomialPrograms should only be created with Models containing Signomial
    Constraints, since Models without Signomials have global solutions and can
    be solved with 'Model.solve()'.""")
        self.gps = []
        self.result = None

    def localsolve(self, solver=None, verbosity=1, x0=None, rel_tol=1e-4,
                   iteration_limit=50, **kwargs):
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
            starttime = time()
        self.gps = []  # NOTE: SIDE EFFECTS
        prevcost, cost, rel_improvement = None, None, None
        while rel_improvement is None or rel_improvement > rel_tol:
            if len(self.gps) > iteration_limit:
                raise RuntimeWarning("""problem unsolved after %s iterations.

    The last result is available in Model.program.gps[-1].result. If the gps
    appear to be converging, you may wish to increase the iteration limit by
    calling .localsolve(..., iteration_limit=NEWLIMIT).""" % len(self.gps))
            gp = self.gp(x0, verbosity=verbosity-1)
            self.gps.append(gp)  # NOTE: SIDE EFFECTS
            try:
                result = gp.solve(solver, verbosity-1, **kwargs)
            except (RuntimeWarning, ValueError):
                nearest_feasible = feasibility_model(gp, "max")
                self.gps.append(nearest_feasible)
                result = nearest_feasible.solve(solver, verbosity=verbosity-1)
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
                  + " and %.3g seconds." % (time() - starttime))

        result["signomialstart"] = startpoint
        self.sens_from_gpconstr(gp.constraints,
                                result["sensitivities"]["constraints"],
                                result["sensitivities"]["constants"])
        self.process_result(result)
        self.result = result  # NOTE: SIDE EFFECTS
        return result

    def gp(self, x0=None, verbosity=1):
        """Get a GP approximation of this SP at x0"""
        gpconstr = self.as_gpconstr(x0)
        return GeometricProgram(self.cost, gpconstr,
                                self.substitutions, verbosity=verbosity)
