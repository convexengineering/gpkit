import numpy as np

from time import time
from functools import reduce as functools_reduce
from operator import add, mul

from .nomials import Posynomial
from .substitution import getsubs
from .small_classes import CootMatrix
from .small_scripts import locate_vars
from .small_scripts import mag

from .geometric_program import GeometricProgram


class SignomialProgram(object):
    """Prepares a collection of signomials for a SP solve.

    Arguments
    ---------
    cost : Constraint
        Signomial to minimize when solving
    constraints : list of Posynomials
        Constraints to maintain when solving (implicitly Signomials <= 1)

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

    def __init__(self, cost, constraints):
        self.cost = cost
        self.constraints = constraints
        self.signomials = [cost] + list(constraints)

        self.posynomials, self.negynomials = [], []
        self.negvarkeys = set()
        for sig in self.signomials:
            p_exps, p_cs = [], []
            n_exps, n_cs = [{}], [1]
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
            raise ValueError("SignomialPrograms must contain coefficients"
                             " less than zero.")

    def solve(self, solver=None, verbosity=1, x0={}, reltol=1e-4,
              iteration_limit=50, *args, **kwargs):
        """Solves a SignomialProgram and returns the solution.

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
        reltol : float
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

        x0 = getsubs({str(vk): vk for vk in self.negvarkeys},
                     self.negvarkeys, x0)
        sp_inits = {vk: vk.descr["sp_init"] for vk in self.negvarkeys
                    if "sp_init" in vk.descr}
        sp_inits.update(x0)
        x0 = sp_inits
        # HACK: initial guess for negative variables
        x0 = {var: 1 for var in self.negvarkeys if var not in x0}

        iterations = 0
        prevcost, cost = 1, 1
        self.gps = []

        while (iterations < iteration_limit
               and (iterations < 2 or
                    abs(prevcost-cost)/(prevcost + cost) > reltol)):
            posy_approxs = []
            abort = -1
            for p, n in zip(self.posynomials, self.negynomials):
                if n is None:
                    posy_approx = p
                else:
                    posy_approx = p/n.mono_approximation(x0)
                posy_approxs.append(posy_approx)

            gp = GeometricProgram(posy_approxs[0], posy_approxs[1:],
                                  verbosity=verbosity-1)
            self.gps.append(gp)
            try:
                result = gp.solve(solver, verbosity=verbosity-1,
                                  *args, **kwargs)
            except (RuntimeWarning, ValueError):
                # TODO: should we add the nearest_feasible gp to the program?
                # TODO: should we count it as an iteration?
                nearest_feasible = gp.feasibility_search(verbosity=verbosity-1)
                result = nearest_feasible.solve(verbosity=verbosity-1)

            x0 = result["variables"]
            prevcost, cost = cost, result["cost"]
            iterations += 1

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
                var_ss = [sens_vars[var]*val for var, val in sens_vars.items()]
                nu_.append(functools_reduce(mul, var_ss, np.sign(c)))
        result["sensitivities"]["monomials"] = np.array(nu_)
        # TODO: SP sensitivities are weird, and potentially incorrect

        return result

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
