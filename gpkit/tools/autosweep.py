"Tools for optimal fits to GP sweeps"
from time import time
import numpy as np
from numpy import log, exp
from gpkit.small_classes import Count
from gpkit.small_scripts import mag


class BinarySweepTree(object):
    def __init__(self, bounds, sols):
        if len(bounds) != 2:
            raise ValueError("bounds must be of length 2.")
        if bounds[1] <= bounds[0]:
            raise ValueError("bounds[0] must be smaller than bounds[1].")
        self.bounds = bounds
        self.sols = sols
        self.costs = log([mag(sol["cost"]) for sol in sols])
        self.splits = None
        self.splitval = None
        self.splitlb = None
        self.splitub = None

    def add_split(self, splitval, splitsol):
        if self.splitval:
            raise ValueError("split already exists!")
        if splitval <= self.bounds[0] or splitval >= self.bounds[1]:
            raise ValueError("split value is at or outside bounds.")
        self.splitval = splitval
        self.splits = [BinarySweepTree([self.bounds[0], splitval],
                                       [self.sols[0], splitsol]),
                       BinarySweepTree([splitval, self.bounds[1]],
                                       [splitsol, self.sols[1]])]

    def add_splitcost(self, splitval, splitlb, splitub):
        if self.splitval:
            raise ValueError("split already exists!")
        if splitval <= self.bounds[0] or splitval >= self.bounds[1]:
            raise ValueError("split value is at or outside bounds.")
        self.splitval = splitval
        self.splitlb, self.splitub = splitlb, splitub

    def var_at_value(self, var, value):
        if value < self.bounds[0] or value > self.bounds[1]:
            raise ValueError("query value is outside bounds.")
        bst = self.min_bst(value)
        lo, hi = bst.bounds
        loval, hival = [sol(var) for sol in bst.sols]
        lo, hi, loval, hival = log(map(mag, [lo, hi, loval, hival]))
        interp = (hi-log(value))/float(hi-lo)
        return exp(interp*loval + (1-interp)*hival)

    def cost_at_value(self, _, value, bound=None):
        if value < self.bounds[0] or value > self.bounds[1]:
            raise ValueError("query value is outside bounds.")
        bst = self.min_bst(value)
        if bst.splitlb:
            if bound:
                if bound is "lb":
                    splitcost = exp(bst.splitlb)
                elif bound is "ub":
                    splitcost = exp(bst.splitub)
            else:
                splitcost = exp((bst.splitlb + bst.splitub)/2)
            if value <= bst.splitval:
                lo, hi = bst.bounds[0], bst.splitval
                loval, hival = bst.sols[0]["cost"], splitcost
            else:
                lo, hi = bst.splitval, bst.bounds[1]
                loval, hival = splitcost, bst.sols[1]["cost"]
        else:
            lo, hi = bst.bounds
            loval, hival = [sol["cost"] for sol in bst.sols]
        lo, hi, loval, hival = log(map(mag, [lo, hi, loval, hival]))
        interp = (hi-log(value))/float(hi-lo)
        return exp(interp*loval + (1-interp)*hival)

    def min_bst(self, value):
        if not self.splits:
            return self
        elif value <= self.splitval:
            return self.splits[0].min_bst(value)
        else:
            return self.splits[1].min_bst(value)

    def __getitem__(self, var):
        return VariableOracle(self, var)


class VariableOracle(object):
    def __init__(self, bst, var):
        self.interpfn = bst.cost_at_value if var is "cost" else bst.var_at_value
        v0 = bst.sols[0]["cost"] if var is "cost" else bst.sols[0](var)
        self.units = getattr(v0, "units", None)
        self.var = var

    def __call__(self, values):
        fit = [self.interpfn(self.var, value) for value in values]
        return fit*self.units if self.units else np.array(fit)

    def lb(self, values):
        fit = [self.interpfn(self.var, value, "lb") for value in values]
        return fit*self.units if self.units else np.array(fit)

    def ub(self, values):
        fit = [self.interpfn(self.var, value, "ub") for value in values]
        return fit*self.units if self.units else np.array(fit)


def autosweep_1d(model, logtol, variable, bounds, **solvekwargs):
    "Autosweep a model over one variable"
    start_time = time()
    solvekwargs.setdefault("verbosity", 1)
    solvekwargs["verbosity"] -= 1
    sols = Count().next
    firstsols = []
    for bound in bounds:
        model.substitutions.update({variable: bound})
        firstsols.append(model.solve(**solvekwargs))
        sols()
    bst = BinarySweepTree(bounds, firstsols)
    tol = recurse_splits(model, bst, variable, logtol, solvekwargs, sols)
    if solvekwargs["verbosity"] > -1:
        print "Solved after %i passes." % sols()
        print "Possible log error +/-%.3g" % tol
        print "Autosweeping took %.3g seconds." % (time() - start_time)
    return bst


def recurse_splits(model, bst, variable, logtol, solvekwargs, sols):
    x, ub, lb = get_tol(bst.costs, bst.bounds, bst.sols, variable)
    tol = (ub-lb)/2.0
    if tol >= logtol:
        model.substitutions.update({variable: x})
        bst.add_split(x, model.solve(**solvekwargs))
        sols()
        tols = [recurse_splits(model, split, variable, logtol, solvekwargs, sols)
                for split in bst.splits]
        return max(tols)
    else:
        bst.add_splitcost(x, lb, ub)
        return tol


def get_tol(costs, bounds, sols, variable):
    y0, y1 = costs
    x0, x1 = log(bounds)
    s0, s1 = [sol["sensitivities"]["constants"][variable] for sol in sols]
    # y0 + s0*(x - x0) == y1 + s1*(x - x1)
    num = y1-y0 + x0*s0-x1*s1
    denom = s0-s1
    if (denom == 0 and num == 0):  # mosek corners
        interp = -1
    else:
        x = num/denom
        lb = y0 + s0*(x-x0)
        interp = (x1-x)/(x1-x0)
        ub = y0*interp + y1*(1-interp)
    if interp < 1e-7 or interp > 1 - 1e-7:  # cvxopt corners
        x = (x0 + x1)/2
        lb = ub = (y0 + y1)/2
    return exp(x), ub, lb
