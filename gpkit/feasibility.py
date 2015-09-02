import numpy as np

from .variables import Variable, VectorVariable
from .varkey import VarKey
from .posyarray import PosyArray


def feasibility_model(model, flavour="max", varname=None, constants=None):
    """Returns a new GP for the closest feasible point of the current GP.

    Arguments
    ---------
    flavour : str
        Specifies the objective function minimized in the search:

        "max" (default) : Apply the same slack to all constraints and
                          minimize that slack. Described in Eqn. 10
                          of [Boyd2007].

        "product" : Apply a unique slack to all constraints and minimize
                    the product of those slacks. Useful for identifying the
                    most problematic constraints. Described in Eqn. 11
                    of [Boyd2007]

    varname : str
        LaTeX name of slack variables.

    *args, **kwargs
        Passed on to GP initialization.

    Returns
    -------
    cost : Posynomial
    constraints : list (of lists) of Signomials

    [Boyd2007] : "A tutorial on geometric programming", Optim Eng 8:67-122

    """

    cost = model.cost
    constraints = model.constraints

    if flavour == "max":
        slackvar = Variable(varname)
        cost = slackvar
        constraints = ([1/slackvar] +  # slackvar > 1
                       [constraint/slackvar  # constraint <= sv
                        for constraint in constraints])
        return cost, constraints

    elif flavour == "product":
        slackvars = VectorVariable(len(constraints), varname)
        cost = np.sum(slackvars)
        constraints = ((1/slackvars).tolist() +  # slackvars > 1
                       [constraint/slackvars[i]  # constraint <= sv
                        for i, constraint in enumerate(constraints)])
        return cost, constraints, slackvars

    elif flavour == "constants":
        if not constants:
            raise ValueError("for 'constants' feasibility analysis, the"
                             " 'constants' argument must be a valid"
                             " substitutions dictionary.")
        slackb = VectorVariable(len(constants))
        constvarkeys, constvars, rmvalue, addvalue = [], [], {}, {}
        for vk in constants.keys():
            descr = dict(vk.descr)
            del descr["value"]
            vk_ = VarKey(**descr)
            rmvalue[vk] = vk_
            addvalue[vk_] = vk
            constvarkeys.append(vk_)
            constvars.append(Variable(**descr))
        constvars = PosyArray(constvars)
        constvalues = PosyArray(constants.values())
        constraints = [c.sub(rmvalue) for c in constraints]
        # cost function could also be .sum(); self.cost would break ties
        cost = slackb.prod()
        constraints = ([slackb >= 1,
                        constvalues/slackb <= constvars,
                        constvars <= constvalues*slackb])
        return cost, constraints, addvalue, constvars, constvarkeys, constvalues
    else:
        raise ValueError("'%s' is not a flavour of feasibility." % flavour)
