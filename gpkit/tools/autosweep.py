"Tools for optimal fits to GP sweeps"
from time import time
import numpy as np
from ..small_classes import Count, Quantity
from ..small_scripts import mag
from ..solution_array import SolutionArray


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
        The left and right logcosts of the segment

    splits : None or two-element list
        If not None, contains the left and right subtrees

    splitval : None or float
        The worst-error point, where the split will be if tolerance is too low

    splitlb : None or float
        The cost lower bound at splitval

    splitub : None or float
        The cost upper bound at splitval
    """

    def __init__(self, bounds, sols, sweptvar, costposy):
        if len(bounds) != 2:
            raise ValueError("bounds must be of length 2.")
        if bounds[1] <= bounds[0]:
            raise ValueError("bounds[0] must be smaller than bounds[1].")
        self.bounds = bounds
        self.sols = sols
        self.costs = np.log([mag(sol["cost"]) for sol in sols])
        self.splits = None
        self.splitval = None
        self.splitlb = None
        self.splitub = None
        self.sweptvar = sweptvar
        self.costposy = costposy

    def add_split(self, splitval, splitsol):
        "Creates subtrees from bounds[0] to splitval and splitval to bounds[1]"
        if self.splitval:
            raise ValueError("split already exists!")
        if splitval <= self.bounds[0] or splitval >= self.bounds[1]:
            raise ValueError("split value is at or outside bounds.")
        self.splitval = splitval
        self.splits = [BinarySweepTree([self.bounds[0], splitval],
                                       [self.sols[0], splitsol],
                                       self.sweptvar, self.costposy),
                       BinarySweepTree([splitval, self.bounds[1]],
                                       [splitsol, self.sols[1]],
                                       self.sweptvar, self.costposy)]

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
        lo, hi, loval, hival = np.log(map(mag, [lo, hi, loval, hival]))
        interp = (hi-np.log(value))/float(hi-lo)
        return np.exp(interp*loval + (1-interp)*hival)

    def cost_at(self, _, value, bound=None):
        "Logspace interpolates between split and costs. Guaranteed bounded."
        if value < self.bounds[0] or value > self.bounds[1]:
            raise ValueError("query value is outside bounds.")
        bst = self.min_bst(value)
        if bst.splitlb:
            if bound:
                if bound is "lb":
                    splitcost = np.exp(bst.splitlb)
                elif bound is "ub":
                    splitcost = np.exp(bst.splitub)
            else:
                splitcost = np.exp((bst.splitlb + bst.splitub)/2)
            if value <= bst.splitval:
                lo, hi = bst.bounds[0], bst.splitval
                loval, hival = bst.sols[0]["cost"], splitcost
            else:
                lo, hi = bst.splitval, bst.bounds[1]
                loval, hival = splitcost, bst.sols[1]["cost"]
        else:
            lo, hi = bst.bounds
            loval, hival = [sol["cost"] for sol in bst.sols]
        lo, hi, loval, hival = np.log(map(mag, [lo, hi, loval, hival]))
        interp = (hi-np.log(value))/float(hi-lo)
        return np.exp(interp*loval + (1-interp)*hival)

    def min_bst(self, value):
        "Returns smallest bst around value."
        if not self.splits:
            return self
        elif value <= self.splitval:
            return self.splits[0].min_bst(value)
        else:
            return self.splits[1].min_bst(value)

    def sample_at(self, values):
        "Creates a SolutionOracle at a given range of values"
        return SolutionOracle(self, values)

    @property
    def sollist(self):
        "Returns a list of all the solutions in an autosweep"
        sollist = [self.sols[0]]
        if self.splits:
            sollist.extend(self.splits[0].sollist[1:])
            sollist.extend(self.splits[1].sollist[1:-1])
        sollist.append(self.sols[1])
        return sollist

    @property
    def solarray(self):
        "Returns a solution array of all the solutions in an autosweep"
        solution = SolutionArray()
        for sol in self.sollist:
            solution.append(sol.program.result)
        solution.to_united_array(unitless_keys=["sensitivities"], united=True)
        units = Quantity(1.0, getattr(self.sols[0]["cost"], "units", None))
        if units:
            solution["cost"] = solution["cost"] * units
        return solution


class SolutionOracle(object):
    "Acts like a SolutionArray for autosweeps"
    def __init__(self, bst, sampled_at):
        self.sampled_at = sampled_at
        self.bst = bst

    def __call__(self, key):
        return self.__getval(key)

    def __getitem__(self, key):
        return self.__getval(key)

    def _is_cost(self, key):
        if hasattr(key, "exps") and (key.exps == self.bst.costposy.exps and
                                     key.cs == self.bst.costposy.cs):
            key = "cost"
        return key is "cost"

    def __getval(self, key):
        "Gets values from the BST and units them"
        if self._is_cost(key):
            key_at = self.bst.cost_at
            v0 = self.bst.sols[0]["cost"]
        else:
            key_at = self.bst.posy_at
            v0 = self.bst.sols[0](key)
        units = Quantity(1.0, getattr(v0, "units", None))
        fit = [key_at(key, x) for x in self.sampled_at]
        return fit*units if units else np.array(fit)

    def cost_lb(self):
        "Gets cost lower bounds from the BST and units them"
        units = Quantity(1.0, getattr(self.bst.sols[0]["cost"], "units", None))
        fit = [self.bst.cost_at("cost", x, "lb") for x in self.sampled_at]
        return fit*units if units else np.array(fit)

    def cost_ub(self):
        "Gets cost upper bounds from the BST and units them"
        units = Quantity(1.0, getattr(self.bst.sols[0]["cost"], "units", None))
        fit = [self.bst.cost_at("cost", x, "ub") for x in self.sampled_at]
        return fit*units if units else np.array(fit)

    def plot(self, posys=None, axes=None):
        "Plots the sweep for each posy"
        import matplotlib.pyplot as plt
        from ..interactive.plot_sweep import assign_axes
        from .. import GPBLU
        if not hasattr(posys, "__len__"):
            posys = [posys]
        for i, posy in enumerate(posys):
            if posy in [None, "cost"]:
                posys[i] = self.bst.costposy
        posys, axes = assign_axes(self.bst.sweptvar, posys, axes)
        for posy, ax in zip(posys, axes):
            if self._is_cost(posy):  # with small tol should look like a line
                ax.fill_between(self.sampled_at,
                                self.cost_lb(), self.cost_ub(),
                                facecolor=GPBLU, edgecolor=GPBLU,
                                linewidth=0.75)
            else:
                ax.plot(self.sampled_at, self(posy), color=GPBLU)
        if len(axes) == 1:
            axes, = axes
        return plt.gcf(), axes

    @property
    def solarray(self):
        "Returns a solution array of all the solutions in an autosweep"
        return self.bst.solarray


def autosweep_1d(model, logtol, sweepvar, bounds, **solvekwargs):
    "Autosweep a model over one sweepvar"
    original_val = model.substitutions.get(sweepvar, None)
    start_time = time()
    solvekwargs.setdefault("verbosity", 1)
    solvekwargs["verbosity"] -= 1
    sols = Count().next
    firstsols = []
    for bound in bounds:
        model.substitutions.update({sweepvar: bound})
        firstsols.append(model.solve(**solvekwargs))
        sols()
    bst = BinarySweepTree(bounds, firstsols, sweepvar, model.cost)
    tol = recurse_splits(model, bst, sweepvar, logtol, solvekwargs, sols)
    bst.nsols = sols()  # pylint: disable=attribute-defined-outside-init
    if solvekwargs["verbosity"] > -1:
        print "Solved after %2i passes, cost logtol +/-%.3g" % (bst.nsols, tol)
        print "Autosweeping took %.3g seconds." % (time() - start_time)
    if original_val:
        model.substitutions[sweepvar] = original_val
    else:
        del model.substitutions[sweepvar]
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
        bst.tol = max(tols)
        return bst.tol
    else:
        bst.add_splitcost(x, lb, ub)
        return tol


# pylint: disable=too-many-locals
def get_tol(costs, bounds, sols, variable):
    "Gets the intersection point and corresponding bounds from two solutions."
    y0, y1 = costs
    x0, x1 = np.log(bounds)
    s0, s1 = [sol["sensitivities"]["constants"][variable] for sol in sols]
    # y0 + s0*(x - x0) == y1 + s1*(x - x1)
    num = y1-y0 + x0*s0-x1*s1
    denom = s0-s1
    # NOTE: several branches below deal with straight lines, where lower
    #       and upper bounds are identical and so x is undefined
    if denom == 0:
        # mosek runs into this on perfect straight lines, num also equal to 0
        # mosek_cli also runs into this on near-straight lines, num ~= 0
        interp = -1  # fflag interp as out-of bounds
    else:
        x = num/denom
        lb = y0 + s0*(x-x0)
        interp = (x1-x)/(x1-x0)
        ub = y0*interp + y1*(1-interp)
    if interp < 1e-7 or interp > 1 - 1e-7:  # cvxopt on straight lines
        x = (x0 + x1)/2  # x is undefined? stick it in the middle!
        lb = ub = (y0 + y1)/2
    return np.exp(x), lb, ub
