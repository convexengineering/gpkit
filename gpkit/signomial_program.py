import numpy as np

from time import time
from functools import reduce as functools_reduce
from operator import add

from .nomials import Posynomial
from .small_classes import CootMatrix
from .small_scripts import locate_vars
from .small_scripts import mag

from .geometric_program import GeometricProgram


class SignomialProgram(object):

    def __init__(self, cost, constraints, printing=True):
        self.cost = cost
        self.constraints = constraints
        signomials = [cost] + list(constraints)

        self.posynomials = []
        self.negynomials = []
        self.negvarkeys = set()
        for sig in signomials:
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
            posy = Posynomial(p_exps, p_cs) if len(p_cs) else None
            negy = Posynomial(n_exps, n_cs) if len(n_cs) > 1 else None
            self.posynomials.append(posy)
            self.negynomials.append(negy)

        if not self.negvarkeys:
            raise ValueError("SignomialPrograms must contain coefficients"
                             "less than zero.")

    def solve(self, solver=None, printing=True, x0={}, reltol=1e-4,
              iteration_limit=50, *args, **kwargs):
        if printing:
            print("Beginning signomial solve.")
            self.starttime = time()

        sp_inits = {vk: vk.descr["sp_init"] for vk in self.negvarkeys
                    if "sp_init" in vk.descr}
        sp_inits.update(x0)
        x0 = sp_inits

        iterations = 0
        prevcost, cost = 1, 1
        self.gps = []

        while (iterations < iteration_limit
               and (iterations < 2 or
                    abs(prevcost-cost)/(prevcost + cost) > reltol)):
            if not x0:
                posy_approxs = self.posynomials
            else:
                posy_approxs = []
                for p, n in zip(self.posynomials, self.negynomials):
                    if n is not None:
                        posy_approxs.append(p/n.mono_approximation(x0))
                    else:
                        posy_approxs.append(p)

            gp = GeometricProgram(posy_approxs[0], posy_approxs[1:])
            self.gps.append(gp)
            try:
                result = gp.solve(solver, printing=printing, *args, **kwargs)
            except RuntimeWarning:
                result = lololol # nearest feasible point

            x0 = result["variables"]
            prevcost = cost
            cost = result["cost"]
            iterations += 1

        if printing:
            print("Solving took %i GP solves" % iterations
                  + " and %.3g seconds." % (time() - self.starttime))

        # need to parse the result and return nu's of original monomials from
        #  variable sensitivities
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
