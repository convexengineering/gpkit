"A module to facilitate testing GPkit against fmincon"
from math import log10, floor
from .. import SignomialsEnabled
from ..small_scripts import mag
# pylint: disable=too-many-statements,too-many-locals

def generate_mfiles(model, algorithm='interior-point', guesstype='ones',
                    gradobj='on', gradconstr='on', writefiles=True):
    """A method for preparing fmincon input files to run a GPkit program

    INPUTS:
        model       [GPkit model] The model to replicate in fmincon

        algorithm:  [string] Algorithm used by fmincon
                    'interior-point': uses the interior point solver
                    'SQP': uses the sequential quadratic programming solver

        guesstype:  [string] The type of initial guess used
                    'ones': One for each variable
                    'order-of-magnitude-floor': The "log-floor" order of
                                                magnitude of the GP/SP optimal
                                                solution (i.e. O(99)=10)
                    'order-of-magnitude-round': The "log-nearest" order of
                                                magnitude of the GP/SP optimal
                                                solution (i.e. O(42)=100)
                    'almost-exact-solution': The GP/SP optimal solution rounded
                                             to 1 significant figure

        gradconstr: [string] Include analytical constraint gradients?
                    'on': Yes
                    'off': No

        gradobj:    [string] Include analytical objective gradients?
                    'on': Yes
                    'off': No

        writefiles: [Boolean] whether or not to actually write the m files
    """

    # Create a new dictionary mapping variables to x(i)'s for use w/ fmincon
    i = 1
    newdict = {}
    lookup = []
    newlist = []
    original_varkeys = model.varkeys
    for key in model.varkeys:
        if key not in model.substitutions:
            newdict[key] = 'x({0})'.format(i)
            newlist += [key.str_without(["units"])]
            lookup += ['x_{0}: '.format(i) + key.str_without(["units"])]
            i += 1
    x0string = make_initial_guess(model, newlist, guesstype)

    cost = model.cost # needs to be before subinplace()
    constraints = model
    constraints.subinplace(constraints.substitutions)
    constraints.subinplace(newdict)

    # Make all constraints less than zero, return list of clean strings
    c = [] # inequality constraints
    ceq = [] # equality constraints
    dc = [] # gradients of inequality constraints
    dceq = [] # gradients of equality constraints
    with SignomialsEnabled():
        for constraint in constraints:
            if constraint.oper == '<=':
                cc = constraint.left - constraint.right
                c += [cc.str_without(["units", "models"])]
            elif constraint.oper == '>=':
                cc = constraint.right - constraint.left
                c += [cc.str_without(["units", "models"])]
            elif constraint.oper == '=':
                cc = constraint.right - constraint.left
                ceq += [cc.str_without(["units", "models"])]

            # Differentiate each constraint w.r.t each variable
            cdm = []
            for key in original_varkeys:
                if key not in model.substitutions:
                    cd = cc.diff(newdict[key])
                    cdm += [cd.str_without("units").replace('**', '.^')]

            if constraint.oper != '=':
                dc += [",...\n          ".join(cdm)]
            else:
                dceq += [",...\n            ".join(cdm)]

    # String for the constraint function .m file
    confunstr = ("function [c, ceq, DC, DCeq] = confun(x)\n" +
                 "% Nonlinear inequality constraints\n" +
                 "c = [\n    " +
                 "\n    ".join(c).replace('**', '.^') +
                 "\n    ];\n\n" +
                 "ceq = [\n      " +
                 "\n      ".join(ceq).replace('**', '.^') +
                 "\n      ];\n" +
                 "if nargout > 2\n    " +
                 "DC = [\n          " +
                 ";\n          ".join(dc) +
                 "\n         ]';\n    " +
                 "DCeq = [\n            " +
                 ";\n            ".join(dceq) +
                 "\n           ]';\n" +
                 "end")

    # Differentiate the objective function w.r.t each variable
    objdiff = []
    for key in original_varkeys:
        if key not in model.substitutions:
            costdiff = cost.diff(key)
            costdiff.subinplace(newdict)
            objdiff += [costdiff.str_without(["units", "models"]).replace('**',
                                                                          '.^')]

    # Replace variables with x(i), make clean string using matlab power syntax
    cost.subinplace(newdict)
    obj = cost.str_without(["units", "models"]).replace('**', '.^')

    # String for the objective function .m file
    objfunstr = ("function [f, gradf] = objfun(x)\n" +
                 "f = " + obj + ";\n" +
                 "if nargout > 1\n" +
                 "    gradf  = [" +
                 "\n              ".join(objdiff) +
                 "];\n" +
                 "end")

    # String for main.m
    mainfunstr = (x0string +
                  "options = optimset('fmincon');\n" +
                  "options.Algorithm = '{0}';\n".format(algorithm) +
                  "options.MaxFunEvals = Inf;\n" +
                  "options.MaxIter = Inf;\n" +
                  "options.GradObj = '{0}';\n".format(gradobj) +
                  "options.GradConstr = '{0}';\n".format(gradconstr) +
                  "tic;\n" +
                  "[x,fval] = ...\n" +
                  "fmincon(@objfun,x0,[],[],[],[],[],[],@confun,options);\n" +
                  "elapsed = toc;\n" +
                  "fid = fopen('elapsed.txt', 'w');\n" +
                  "fprintf(fid, '%.1f', elapsed);\n" +
                  "fclose(fid);\n" +
                  "fid = fopen('cost.txt', 'w');\n" +
                  "fprintf(fid, '%.5g', fval);\n" +
                  "fclose(fid);")

    if writefiles:
        # Write the constraint function .m file
        with open('confun.m', 'w') as outfile:
            outfile.write(confunstr)

        # Write the objective function .m file
        with open('objfun.m', 'w') as outfile:
            outfile.write(objfunstr)

        # Write a txt file for looking up original variable names
        with open('lookup.txt', 'w') as outfile:
            outfile.write("\n".join(lookup))

        # Write the main .m file for running fmincon
        with open('main.m', 'w') as outfile:
            outfile.write(mainfunstr)

    return obj, c, ceq, dc, dceq


def make_initial_guess(model, newlist, guesstype='ones'):
    """Returns initial guess"""
    try:
        sol = model.solve(verbosity=0)
    except TypeError:
        sol = model.localsolve(verbosity=0)
    if guesstype == "ones":
        x0string = ["x0 = ones({0},1);\n".format(len(sol['freevariables']))]
    else:
        x0string = ["x0 = ["]
        i = 1
        for vk in newlist:
            xf = mag(sol['freevariables'][vk])
            if guesstype == "almost-exact-solution":
                x0 = round(xf, -int(floor(log10(abs(xf))))) # rounds to 1sf
            elif guesstype == "order-of-magnitude-floor":
                x0 = 10**floor(log10(xf))
            elif guesstype == "order-of-magnitude-round":
                x0 = 10**round(log10(xf))
            else:
                raise Exception("Unexpected guess type")
            x0string += [str(x0) + ", "]
            i += 1
        x0string += ["];\n"]

    return "".join(x0string)
