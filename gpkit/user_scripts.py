import numpy as np

from collections import Iterable

from .nomials import Posynomial
from .variables import Variable, VectorVariable
from .posyarray import PosyArray
from .geometric_program import GeometricProgram


def make_feasibility_gp(gp, varname=None, flavour="max"):
    if flavour == "max":
        slackvar = Variable(varname)
        gp_ = GeometricProgram(slackvar,
                               [constraint <= slackvar
                                for constraint in gp.constraints],
                               substitutions=gp.substitutions)
    elif flavour == "product":
        slackvars = VectorVariable(len(gp.constraints), varname)
        gp_ = GeometricProgram(slackvars.prod(),
                               [constraint <= slackvars[i]
                                for i, constraint in enumerate(gp.constraints)],
                               substitutions=gp.substitutions)
    else:
        raise ValueError("'%s' is an unknown flavour of feasibility." % flavour)
    return gp_


def find_feasible_point(gp, flavour="max", *args, **kwargs):
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
