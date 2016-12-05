"Tools for optimal fits to GP sweeps"
import numpy as np
from numpy import log, exp
from gpkit.small_classes import Count
from gpkit.small_scripts import mag


def assert_logtol(x, y, logtol=1e-6):
    np.testing.assert_allclose(log(mag(x)), log(mag(y)), atol=logtol, rtol=0)


class BinarySweepTree(object):
    def __init__(self, bounds, sols):
        if len(bounds) != 2:
            raise ValueError("bounds must be of length 2.")
        if bounds[1] <= bounds[0]:
            raise ValueError("bounds[0] must be smaller than bounds[1].")
        self.bounds = bounds
        self.sols = sols
        self.costs = [sol["cost"] for sol in sols]
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
        if self.units:
            fit = fit * self.units
        else:
            fit = np.array(fit)
        return fit

    def lb(self, values):
        fit = [self.interpfn(self.var, value, "lb") for value in values]
        if self.units:
            fit = fit * self.units
        else:
            fit = np.array(fit)
        return fit

    def ub(self, values):
        fit = [self.interpfn(self.var, value, "ub") for value in values]
        if self.units:
            fit = fit * self.units
        else:
            fit = np.array(fit)
        return fit

bst0 = BinarySweepTree([1, 2], [{"cost": 1}, {"cost": 8}])
assert_logtol(bst0["cost"]([1, 1.5, 2]), [1, 3.375, 8], 1e-3)
bst0.add_split(1.5, {"cost": 4})
assert_logtol(bst0["cost"]([1, 1.25, 1.5, 1.75, 2]),
              [1, 2.144, 4, 5.799, 8], 1e-3)


def sweep_1d(model, logtol, variable, bounds, conservative=False, **solvekwargs):
    "Autosweep a model over one variable"
    bst = BinarySweepTree(bounds, [])
    solvekwargs.setdefault("verbosity", 1)
    solvekwargs["verbosity"] -= 1
    sols = Count().next
    for bound in bounds:
        model.substitutions.update({variable: bound})
        bst.sols.append(model.solve(**solvekwargs))
        sols()
    tol = recurse_splits(model, bst, variable, logtol, solvekwargs, sols)
    print "Fit with max log error of %.3g after %i solutions. " % (tol, sols())
    return bst


def recurse_splits(model, bst, variable, logtol, solvekwargs, sols):
    x, ub, lb = get_tol(bst.sols, variable)
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


def get_tol(sols, variable):
    y0, y1 = [log(mag(sol["cost"])) for sol in sols]
    x0, x1 = [log(mag(sol["constants"][variable])) for sol in sols]
    senss = [sol["sensitivities"]["constants"][variable] for sol in sols]
    # y0 + senss[0]*(x - x0) == y1 + senss[1]*(x - x1)
    x = (y1-y0 + x0*senss[0]-x1*senss[1])/(senss[0]-senss[1])
    lb = y0 + senss[0]*(x-x0)
    interp = (x1-x)/(x1-x0)
    ub = y0*interp + y1*(1-interp)
    return exp(x), ub, lb

from gpkit import *
x = Variable("x", "m**2")
xmin = Variable("xmin", "m")
m = Model(x**2, [x >= xmin**2 + units.m**2])

xmin_ = np.linspace(1, 10, 100)
for dec in range(6):
    tol = 10**-dec
    print "Testing with tolerance of %.3g" % tol
    bst = sweep_1d(m, tol, xmin, [1, 10])
    assert_logtol(bst["xmin"](xmin_), xmin_)
    assert_logtol(bst["x"](xmin_), xmin_**2 + 1, tol)
    assert_logtol(bst["cost"](xmin_), (xmin_**2 + 1)**2, tol)

assert bst["cost"](xmin_).units == ureg.m**4
assert bst["x"](xmin_).units == ureg.m**2

bst = sweep_1d(m, 1, xmin, [1, 10])
# fill_between(xmin_, bst["cost"].lb(xmin_), bst["cost"].ub(xmin_))
