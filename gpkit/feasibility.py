import numpy as np

from .variables import Variable, VectorVariable
from .varkey import VarKey
from .posyarray import PosyArray


def feasibility_model(program, flavour="max", varname=None, constants=None):
    """Returns a new GP for the closest feasible point of the current GP.

    Arguments
    ---------
    flavour : str
        Specifies the objective function minimized in the search:

        "max" (default) : Apply the same slack to all constraints and
                          minimize that slack. Described in Eqn. 10
                          of [Boyd2007].

        "product" : Apply a unique slack to all constraints and minimize
                    the product of those slack variables.. Useful for
                    identifying the most problematic constraints. Described in
                    Eqn. 11 of [Boyd2007]

        "constants" : Slack the constants of a problem and minimize the product
                      of those slack variables.

    varname : str
        LaTeX name of slack variables.

    Returns
    -------
    program : Program of the same type as the input
              (Model, GeometricProgram, or SignomialProgram)

    [Boyd2007] : "A tutorial on geometric programming", Optim Eng 8:67-122

    """

    cost = program.cost
    constraints = program.constraints
    programType = program.__class__

    if flavour == "max":
        slackvar = Variable(varname)
        cost = slackvar
        constraints = ([1/slackvar] +  # slackvar > 1
                       [constraint/slackvar  # constraint <= sv
                        for constraint in constraints])
        prog = programType(cost, constraints)

    elif flavour == "product":
        slackvars = VectorVariable(len(constraints), varname)
        cost = np.sum(slackvars)
        constraints = ((1/slackvars).tolist() +  # slackvars > 1
                       [constraint/slackvars[i]  # constraint <= sv
                        for i, constraint in enumerate(constraints)])
        prog = programType(cost, constraints)
        prog.slackvars = slackvars

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
        prog = programType(cost, constraints)
        prog.addvalue = addvalue
        prog.constvars = constvars
        prog.constvarkeys = constvarkeys
        prog.constvalues = constvalues

    else:
        raise ValueError("'%s' is not a flavour of feasibility." % flavour)

    return prog
