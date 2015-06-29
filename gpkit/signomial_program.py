import numpy as np

from time import time
from functools import reduce as functools_reduce
from operator import add

from .small_classes import CootMatrix
from .small_scripts import locate_vars
from .small_scripts import mag

from .geometric_program import GeometricProgram


class SignomialProgram(GeometricProgram):

    def __init__(self, cost, constraints, printing=True):
        self.cost = cost
        self.constraints = constraints
        signomials = [cost] + list(constraints)

        self.posynomials = [sum([0]+[m for m in sig if m.c > 0])
                            for sig in signomials]

        self.negynomials = [sum([0]+[-m for m in sig if m.c < 0])
                            for sig in signomials]

        if all([all([c > 0 for m in sig]) for sig in signomials]):
            raise ValueError("SignomialPrograms must contain coefficients"
                             "less than zero.")

    def solve(self, solver=None, printing=True, x0={}, reltol=1e-4,
              iteration_limit=50, *args, **kwargs):
        if printing:
            print("Beginning signomial solve.")
            self.starttime = time()
        
        iterations = 0
        prevcost, cost = 1, 1

        vk_inits = {vk: vk.descr["sp_init"] for vk in neg_varkeys
                    if "sp_init" in vk.descr}
        vk_inits.update(x0)
        x0 = vk_inits

        self.gps = []

        while (iterations < iteration_limit
               and (iterations < 2 or
                    abs(prevcost-cost)/(prevcost + cost) > self.reltol)):
            if not x0:
                posy_approxs = self.posynomials
            else:
                posy_approxs = [p/(1+n).mono_approximation(x0) for p, n
                                in zip(self.posynomials, self.negynomials)]

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

        return result
