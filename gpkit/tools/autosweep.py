"Tools for optimal fits to GP sweeps"
from time import time
import numpy as np
from numpy import log, exp
from gpkit.small_classes import Count
from gpkit.small_scripts import mag


# pylint: disable=too-many-instance-attributes
class BinarySweepTree(object):
    """Spans a line segment. May contain two subtrees that divide the segment.

    Attributes
    ----------

    bounds : two-element list
        The left and right boundaries of the segment

    sols : two-element list
        The left and right solutions of the segment

    costs : array
        The left and right logcosts of the span

    splits : None or two-element list
        If not None, contains the left and right subtrees

    splitval : None or float
        The worst-error point, where the split will be if tolerance is too low

    splitlb : None or float
        The cost lower bound at splitval

    splitub : None or float
        The cost upper bound at splitval
    """

    def __init__(self, bounds, sols, cost_posy=None):
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
        self.cost_posy = cost_posy

    def add_split(self, splitval, splitsol):
        "Creates subtrees from bounds[0] to splitval and splitval to bounds[1]"
        if self.splitval:
            raise ValueError("split already exists!")
        if splitval <= self.bounds[0] or splitval >= self.bounds[1]:
            raise ValueError("split value is at or outside bounds.")
        self.splitval = splitval
        self.splits = [BinarySweepTree([self.bounds[0], splitval],
                                       [self.sols[0], splitsol],
                                       self.cost_posy),
                       BinarySweepTree([splitval, self.bounds[1]],
                                       [splitsol, self.sols[1]],
                                       self.cost_posy)]

    def add_splitcost(self, splitval, splitlb, splitub):
        "Adds a splitval, lower bound, and upper bound"
        if self.splitval:
            raise ValueError("split already exists!")
        if splitval <= self.bounds[0] or splitval >= self.bounds[1]:
            raise ValueError("split value is at or outside bounds.")
        self.splitval = splitval
        self.splitlb, self.splitub = splitlb, splitub

    def posy_at(self, posy, value):
        """Logspace interpolates between sols to get posynomial values.

        No guarantees, just like a regular sweep.
        """
        if value < self.bounds[0] or value > self.bounds[1]:
            raise ValueError("query value is outside bounds.")
        bst = self.min_bst(value)
        lo, hi = bst.bounds
        loval, hival = [sol(posy) for sol in bst.sols]
        lo, hi, loval, hival = log(map(mag, [lo, hi, loval, hival]))
        interp = (hi-log(value))/float(hi-lo)
        return exp(interp*loval + (1-interp)*hival)

    def cost_at(self, _, value, bound=None):
        "Logspace interpolates between split and costs. Guaranteed bounded."
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
        "Returns smallest spanning tree around value."
        if not self.splits:
            return self
        elif value <= self.splitval:
            return self.splits[0].min_bst(value)
        else:
            return self.splits[1].min_bst(value)

    def sample_at(self, values):
        "Creates a SolutionOracle at a given range of values"
        return SolutionOracle(self, values)


class SolutionOracle(object):
    "Acts like a SolutionArray for autosweeps"
    def __init__(self, bst, values):
        self.values = values
        self.bst = bst

    def __call__(self, key):
        return self.__getval(key)

    def __getitem__(self, key):
        return self.__getval(key)

    def __getval(self, key):
        "Gets values from the BST and units them"
        if hasattr(key, "exps") and (key.exps == self.bst.cost_posy.exps and
                                     key.cs == self.bst.cost_posy.cs):
            key = "cost"
        if key is "cost":
            key_at = self.bst.cost_at
            v0 = self.bst.sols[0]["cost"]
        else:
            key_at = self.bst.posy_at
            v0 = self.bst.sols[0](key)
        units = getattr(v0, "units", None)
        fit = [key_at(key, value) for value in self.values]
        return fit*units if units else np.array(fit)

    def cost_lb(self):
        "Gets cost lower bounds from the BST and units them"
        units = getattr(self.bst.sols[0]["cost"], "units", None)
        fit = [self.bst.cost_at("cost", value, "lb") for value in self.values]
        return fit*units if units else np.array(fit)

    def cost_ub(self):
        "Gets cost upper bounds from the BST and units them"
        units = getattr(self.bst.sols[0]["cost"], "units", None)
        fit = [self.bst.cost_at("cost", value, "ub") for value in self.values]
        return fit*units if units else np.array(fit)


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
    bst = BinarySweepTree(bounds, firstsols, model.cost)
    tol = recurse_splits(model, bst, variable, logtol, solvekwargs, sols)
    if solvekwargs["verbosity"] > -1:
        print "Solved after %i passes, possible logerr +/-%.3g" % (sols(), tol)
    if solvekwargs["verbosity"] > 0:
        print "Autosweeping took %.3g seconds." % (time() - start_time)
    return bst


def recurse_splits(model, bst, variable, logtol, solvekwargs, sols):
    "Recursively splits a BST until logtol is reached"
    x, lb, ub = get_tol(bst.costs, bst.bounds, bst.sols, variable)
    tol = (ub-lb)/2.0
    if tol >= logtol:
        model.substitutions.update({variable: x})
        bst.add_split(x, model.solve(**solvekwargs))
        sols()
        tols = [recurse_splits(model, split, variable, logtol, solvekwargs,
                               sols)
                for split in bst.splits]
        return max(tols)
    else:
        bst.add_splitcost(x, lb, ub)
        return tol


# pylint: disable=too-many-locals
def get_tol(costs, bounds, sols, variable):
    "Gets the intersection point and corresponding bounds from two solutions."
    y0, y1 = costs
    x0, x1 = log(bounds)
    s0, s1 = [sol["sensitivities"]["constants"][variable] for sol in sols]
    # y0 + s0*(x - x0) == y1 + s1*(x - x1)
    num = y1-y0 + x0*s0-x1*s1
    denom = s0-s1
    if denom == 0:
        # mosek only runs into this on perfectly corners, where num == 0
        # mosek_cli seems to run into this on non-sharp corners...
        interp = -1
    else:
        x = num/denom
        lb = y0 + s0*(x-x0)
        interp = (x1-x)/(x1-x0)
        ub = y0*interp + y1*(1-interp)
    if interp < 1e-7 or interp > 1 - 1e-7:  # cvxopt corners
        x = (x0 + x1)/2
        lb = ub = (y0 + y1)/2
    return exp(x), lb, ub
