import numpy as np

from .variables import Variable, VectorVariable
from .varkey import VarKey
from .posyarray import PosyArray


def feasibility_model(program, flavour="max", varname=None,
                      constants=None, signomials=None, programType=None):
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
    if not programType:
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
        slackb = VectorVariable(len(constants), units="-")
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
        constraints = [c.sub(rmvalue) for c in signomials]
        cost = slackb.prod()
        # cost function could also be .sum(); self.cost would break ties
        for i in range(len(constvars)):
            slack = slackb[i]
            constvar, constvalue = constvars[i], constvalues[i]
            if constvar.units:
                constvalue *= constvar.units
            constraints += [slack >= 1,
                            constvalue/slack <= constvar,
                            constvar <= constvalue*slack]
        prog = programType(cost, constraints)
        prog.addvalue = addvalue
        prog.constvars = constvars
        prog.constvarkeys = constvarkeys
        prog.constvalues = constvalues
        prog.slackb = slackb

    else:
        raise ValueError("'%s' is not a flavour of feasibility." % flavour)

    return prog
