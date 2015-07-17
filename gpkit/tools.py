"""Non-application-specific convenience methods for GPkit"""
import numpy as np

from collections import Iterable

from .variables import Variable, VectorVariable
from .posyarray import PosyArray
from .geometric_program import GeometricProgram


def te_exp_minus1(posy, nterm):
    """Taylor expansion of e^{posy} - 1

    Arguments
    ---------
    posy : gpkit.Posynomial
        Variable or expression to exponentiate
    nterm : int
        Number of terms in resulting Taylor expansion

    Returns
    -------
    gpkit.Posynomial
        Taylor expansion of e^{posy} - 1, carried to nterm terms
    """
    if nterm < 1:
        raise ValueError("Unexpected number of terms, nterm=%s" % nterm)
    res = 0
    factorial_denom = 1
    for i in xrange(1, nterm + 1):
        factorial_denom *= i
        res += posy**i / factorial_denom
    return res


def make_feasibility_gp(gp, varname=None, flavour="max"):
    """Given a GP, returns a feasible GP.

    "Flavour" specifies the objective function minimized in the search for
    feasibility:

        "max" (default) : Apply the same slack to all constraints and minimize
                          that slack. Useful for finding the "closest"
                          feasible point. Described in Eqn. 10 of [Boyd2007].

        "product" : Apply a unique slack to all constraints and minimize the
                    product of those slacks. Useful for identifying the most
                    problematic constraints. Described in Eqn. 11 of [Boyd2007]


    [Boyd2007] : "A tutorial on geometric programming", Optim Eng 8:67-122

    """

    if flavour == "max":
        slackvar = Variable(varname)
        gp_ = GeometricProgram(slackvar,
                               [slackvar >= 1] +
                               [constraint <= slackvar
                                for constraint in gp.constraints],
                               substitutions=gp.substitutions)
    elif flavour == "product":
        slackvars = VectorVariable(len(gp.constraints), varname)
        gp_ = GeometricProgram(slackvars.prod(),
                               [slackvars >= 1] +
                               [constraint <= slackvars[i]
                                for i, constraint in enumerate(gp.constraints)],
                               substitutions=gp.substitutions)
    else:
        raise ValueError("'%s' is an unknown flavour of feasibility." % flavour)
    return gp_


def closest_feasible_point(gp, flavour="max", *args, **kwargs):
    return make_feasibility_gp(gp, flavour=flavour).solve(*args, **kwargs)


def composite_objective(*objectives, **kwargs):
    objectives = list(objectives)
    n = len(objectives)
    if "k" in kwargs:
        k = kwargs["k"]
    else:
        k = 10
    if "sweep" in kwargs:
        sweeps = [kwargs["sweep"]]*(n-1)
    elif "sweeps" in kwargs:
        sweeps = kwargs["sweeps"]
    else:
        sweeps = [np.linspace(0, 1, k)]*(n-1)
    if "normsub" in kwargs:
        normalization = [p.sub(kwargs["normsub"]) for p in objectives]
    else:
        normalization = [1]*n

    sweeps = list(zip(["sweep"]*(n-1), sweeps))
    ws = VectorVariable(n-1, "w_{CO}", sweeps, "-")
    w_s = []
    for w in ws:
        descr = dict(w.descr)
        del descr["value"]
        descr["name"] = "v_{CO}"
        w_s.append(Variable(value=('sweep', lambda x: 1-x), args=[w], **descr))
    w_s = normalization[-1]*PosyArray(w_s)*objectives[-1]
    objective = w_s.prod()
    for i, obj in enumerate(objectives[:-1]):
        objective += ws[i]*w_s[:i].prod()*w_s[i+1:].prod()*obj/normalization[i]
    return objective


def link(gps, varids):
    if not isinstance(gps, Iterable):
        gps = [gps]
    if not isinstance(varids, Iterable):
        varids = [varids]

    if isinstance(varids, dict):
        subs = {getvarstr(k): getvarkey(v) for k, v in varids.items()}
    else:
        subs = {getvarstr(v): getvarkey(v) for v in varids}

    for gp in gps:
        gp.sub(subs)

    gppile = gps[0]
    for gp in gps[1:]:
        gppile += gp
    return gppile


def getvarkey(var):
    if isinstance(var, str):
        return gps[0].varkeys[var]
    else:
        # assume is VarKey or Monomial
        return var


def getvarstr(var):
    if isinstance(var, str):
        return var
    else:
        # assume is VarKey or Monomial
        if hasattr(var, "_cmpstr"):
            return var._cmpstr
        else:
            return list(var.exp)[0]._cmpstr
