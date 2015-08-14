"""Implement the SignomialProgram class"""
import numpy as np

from time import time
from functools import reduce as functools_reduce
from operator import mul

from .nomials import Posynomial
from .geometric_program import GeometricProgram

from .substitution import get_constants


class SignomialProgram(object):
    """Prepares a collection of signomials for a SP solve.

    Arguments
    ---------
    cost : Constraint
        Signomial to minimize when solving
    constraints : list of Signomials
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

        self.posynomials, self.negynomials = [self.cost], [None]
        self.negvarkeys = set()
        for sig in self.constraints:
            p_exps, p_cs = [], []
            n_exps, n_cs = [{}], [1]  # add the 1 from the "<= 1" constraint
            for c, exp in zip(sig.cs, sig.exps):
                if c > 0:
                    p_cs.append(c)
                    p_exps.append(exp)
                elif c < 0:
                    n_cs.append(-c)
                    n_exps.append(exp)
                    self.negvarkeys.update(exp.keys())
            posy = Posynomial(p_exps, p_cs) if p_cs != [] else None
            negy = Posynomial(n_exps, n_cs) if n_cs != [1] else None
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

        if x0 is None:
            x0 = {}
        else:
            # dummy nomial data to turn x0's keys into VarKeys
            self.negydata = lambda: None
            self.negydata.varlocs = self.negvarkeys
            self.negydata.varstrs = {str(vk): vk for vk in self.negvarkeys}
            x0 = get_constants(self.negydata, x0)
        sp_inits = {vk: vk.descr["sp_init"] for vk in self.negvarkeys
                    if "sp_init" in vk.descr}
        sp_inits.update(x0)
        x0 = sp_inits
        # HACK: initial guess for negative variables
        x0.update({var: 1 for var in self.negvarkeys if var not in x0})

        iterations = 0
        prevcost, cost, rel_improvement = None, None, None
        self.gps = []

        while rel_improvement is None or rel_improvement > rel_tol:
            if iterations > iteration_limit:
                raise RuntimeWarning("""problem unsolved after %s iterations.

    The last result is available in Model.program.gps[-1].result. If the gps
    appear to be converging, you may wish to increase the iteration limit by
    calling .localsolve(..., iteration_limit=NEWLIMIT).""" % iterations)

            gp = self.step(x0, verbosity)
            self.gps.append(gp)  # NOTE: SIDE EFFECTS

            try:
                result = gp.solve(solver, verbosity=verbosity-1,
                                  *args, **kwargs)
            except (RuntimeWarning, ValueError):
                # TODO: should we add the nearest_feasible gp to the program?
                # TODO: should we count it as an iteration?
                nearest_feasible = gp.feasibility_search(verbosity=verbosity-1)
                result = nearest_feasible.solve(verbosity=verbosity-1)
                result["cost"] = None

            x0 = result["variables"]
            prevcost, cost = cost, result["cost"]
            if cost and prevcost:
                rel_improvement = abs(prevcost-cost)/(prevcost + cost)

            iterations += 1

        # solved successfully!
        if verbosity > 0:
            print("Solving took %i GP solves" % iterations
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
            # HACK: initial guess for negative variables
            x0 = {var: 1 for var in self.negvarkeys}
            sp_inits = {vk: vk.descr["sp_init"] for vk in self.negvarkeys
                        if "sp_init" in vk.descr}
            x0,update(sp_inits)
        posy_approxs = []
        for p, n in zip(self.posynomials, self.negynomials):
            if n is None:
                posy_approx = p
            else:
                posy_approx = p/n.mono_approximation(x0)
            posy_approxs.append(posy_approx)

        gp = GeometricProgram(posy_approxs[0], posy_approxs[1:],
                              verbosity=verbosity-1)
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
