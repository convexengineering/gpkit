"""Implement the SignomialProgram class"""
import numpy as np

from time import time
from functools import reduce as functools_reduce
from operator import mul

from .geometric_program import GeometricProgram

from .substitution import get_constants
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

    def __init__(self, cost, constraints, verbosity=2):
        if cost.any_nonpositive_cs:
            raise TypeError("""SignomialPrograms need Posyomial objectives.

    The equivalent of a Signomial objective can be constructed by constraining
    a dummy variable z to be greater than the desired Signomial objective s
    (z >= s) and then minimizing that dummy variable.""")

        self.cost = cost
        self.constraints = constraints
        self.signomials = [cost] + list(constraints)

        self.posynomials, self.negynomials = [self.cost], [0]
        self.negvarkeys = set()
        for sig in self.constraints:
            if not sig.any_nonpositive_cs:
                posy, negy = sig, 0
            else:
                posy, negy = sig.posy_negy()
                if len(negy.cs) == 1:
                    raise ValueError("Signomial constraint has only one"
                                     " negative monomial; it should have been"
                                     " a Posynomial constraint.")
                self.negvarkeys.update(negy.varlocs)
            self.posynomials.append(posy)
            self.negynomials.append(negy)

        if not self.negvarkeys:
            raise ValueError("SignomialPrograms must contain at least one"
                             " Signomial.")

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

        # parse the result and return nu's of original monomials from
        #  variable sensitivities
        nu = result["sensitivities"]["monomials"]
        sens_vars = {var: sum([gp.exps[i][var]*nu[i] for i in locs])
                     for (var, locs) in gp.varlocs.items()}
        nu_ = []
        for signomial in self.signomials:
            for c, exp in zip(signomial.cs, signomial.exps):
                var_ss = [sens_vars[var]*val for var, val in exp.items()]
                nu_.append(functools_reduce(mul, var_ss, np.sign(c)))
        result["sensitivities"]["monomials"] = np.array(nu_)
        # TODO: SP sensitivities are weird, and potentially incorrect

        self.result = result  # NOTE: SIDE EFFECTS
        return result

    def step(self, x0=None, verbosity=1):
        if x0 is None:
            # dummy nomial data to turn x0's keys into VarKeys
            self.negydata = lambda: None
            self.negydata.varlocs = self.negvarkeys
            self.negydata.varstrs = {str(vk): vk for vk in self.negvarkeys}
            x0 = get_constants(self.negydata, {})
            sp_inits = {vk: vk.descr["sp_init"] for vk in self.negvarkeys
                        if "sp_init" in vk.descr}
            x0.update(sp_inits)
            # HACK: initial guess for negative variables
            x0.update({var: 1 for var in self.negvarkeys if var not in x0})
        posy_approxs = []
        for p, n in zip(self.posynomials, self.negynomials):
            if n is 0:
                posy_approx = p
            else:
                posy_approx = p/n.mono_lower_bound(x0)
            posy_approxs.append(posy_approx)

        gp = GeometricProgram(posy_approxs[0], posy_approxs[1:],
                              verbosity=verbosity)
        return gp

    def __repr__(self):
        return "gpkit.%s(\n%s)" % (self.__class__.__name__, str(self))

    def __str__(self):
        """String representation of a SignomialProgram.

        Contains all of its parameters."""
        return "\n".join(["  # minimize",
                          "    %s," % self.cost,
                          "[ # subject to"] +
                         ["    %s <= 1," % constr
                          for constr in self.constraints] +
                         [']'])

    def _latex(self, unused=None):
        """LaTeX representation of a SignomialProgram.

        Contains all of its parameters."""
        return "\n".join(["\\begin{array}[ll]",
                          "\\text{}",
                          "\\text{minimize}",
                          "    & %s \\\\" % self.cost._latex(),
                          "\\text{subject to}"] +
                         ["    & %s \\leq 1\\\\" % constr._latex()
                          for constr in self.constraints] +
                         ["\\end{array}"])
