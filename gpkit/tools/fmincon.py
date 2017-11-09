"A module to facilitate testing GPkit against fmincon"
from math import log10, floor, log
from .. import SignomialsEnabled
from ..varkey import VarKey
from ..keydict import KeyDict
from ..small_scripts import mag
# pylint: disable=too-many-statements,too-many-locals,too-many-branches

def generate_mfiles(model, logspace=False, algorithm='interior-point',
                    guess='ones', gradobj='on', gradconstr='on',
                    writefiles=True):
    """A method for preparing fmincon input files to run a GPkit program

    INPUTS:
        model       [GPkit model] The model to replicate in fmincon

        logspace    [Boolean] Whether to re-produce the model in logspace

        algorithm:  [string] Algorithm used by fmincon
                    'interior-point': uses the interior point solver
                    'SQP': uses the sequential quadratic programming solver

        guess:      [string] The type of initial guess used
                    'ones': One for each variable
                    'order-of-magnitude-floor': The "log-floor" order of
                                                magnitude of the GP/SP optimal
                                                solution (i.e. O(99)=10)
                    'order-of-magnitude-round': The "log-nearest" order of
                                                magnitude of the GP/SP optimal
                                                solution (i.e. O(42)=100)
                    'almost-exact-solution': The GP/SP optimal solution rounded
                                             to 1 significant figure
                    OR
                    [list] The actual values of initial guess to use

        gradconstr: [string] Include analytical constraint gradients?
                    'on': Yes
                    'off': No

        gradobj:    [string] Include analytical objective gradients?
                    'on': Yes
                    'off': No

        writefiles: [Boolean] whether or not to actually write the m files
    """
    if logspace: # Supplying derivatives not supported for logspace
        gradobj = 'off'
        gradconstr = 'off'

    # Create a new dictionary mapping variables to x(i)'s for use w/ fmincon
    i = 1
    newdict = {}
    lookup = []
    newlist = []
    original_varkeys = model.varkeys
    for key in model.varkeys:
        if key not in model.substitutions:
            descr = key.descr.copy()
            descr["name"] = 'x(%i)' % i
            newdict[key] = VarKey(**descr)
            newlist += [key.str_without(["units"])]
            lookup += ['x_{0}: '.format(i) + key.str_without(["units"])]
            i += 1
    x0string = make_initial_guess(model, newlist, guess, logspace)

    cost = model.cost # needs to be before subinplace()
    constraints = model
    substitutions = constraints.substitutions
    constraints.substitutions = KeyDict()
    constraints.subinplace(substitutions)
    constraints.subinplace(newdict)
    constraints.substitutions = substitutions

    # Make all constraints less than zero, return list of clean strings
    c = [] # inequality constraints
    ceq = [] # equality constraints
    dc = [] # gradients of inequality constraints
    dceq = [] # gradients of equality constraints

    if logspace:
        for constraint in constraints:
            expdicttuple = constraint.as_posyslt1()[0].exps
            clist = mag(constraint.as_posyslt1()[0].cs)

            constraintstring = ['log(']
            for expdict, C in zip(expdicttuple, clist):
                constraintstring += ['+ {0}*exp('.format(C)]
                for k, v in expdict.iteritems():
                    constraintstring += ['+{0} * {1}'.format(v, k)]
                constraintstring += [')']
            constraintstring += [')']

            if constraint.oper == '=':
                ceq += [' '.join(constraintstring)]
            else:
                c += [' '.join(constraintstring)]
    else:
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

    # Objective function (and derivatives if applicable)
    cost.subinplace(newdict)
    objdiff = []
    if logspace:
        objstring = ['log(']
        expdicttuple = cost.exps
        clist = mag(cost.cs)
        for expdict, cc in zip(expdicttuple, clist):
            objstring += ['+ {0}*exp('.format(cc)]
            for k, v in expdict.iteritems():
                objstring += ['+{0} * {1}'.format(v, k)]
            objstring += [')']
        objstring += [')']
        obj = ' '.join(objstring)
    else:
        # Differentiate the objective function w.r.t each variable
        for key in original_varkeys:
            if key not in model.substitutions:
                costdiff = cost.diff(key)
                costdiff.subinplace(newdict)
                objdiff += [costdiff.str_without(["units", "models"]).replace(
                    '**', '.^')]

        # Replace variables with x(i), make clean string using matlab power syn.
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
    fval = "exp(fval)" if logspace else "fval"
    mainfunstr = (x0string +
                  "options = optimset('fmincon');\n" +
                  "options.Algorithm = '{0}';\n".format(algorithm) +
                  "options.MaxFunEvals = Inf;\n" +
                  "options.MaxIter = Inf;\n" +
                  "options.GradObj = '{0}';\n".format(gradobj) +
                  "options.GradConstr = '{0}';\n".format(gradconstr) +
                  "tic;\n" +
                  "[x,fval, exitflag, output] = ...\n" +
                  "fmincon(@objfun,x0,[],[],[],[],[],[],@confun,options);\n" +
                  "elapsed = toc;\n" +
                  "fid = fopen('elapsed.txt', 'w');\n" +
                  "fprintf(fid, '%.1f', elapsed);\n" +
                  "fclose(fid);\n" +
                  "fid = fopen('iterations.txt', 'w');\n" +
                  "fprintf(fid, '%d', output.iterations);\n" +
                  "fclose(fid);\n" +
                  "fid = fopen('cost.txt', 'w');\n" +
                  "fprintf(fid, '%.5g', {0});\n".format(fval) +
                  "if exitflag == -2\n\tfprintf(fid, '(i)');\nend\n" +
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


def make_initial_guess(model, newlist, guess='ones', logspace=False):
    """Returns initial guess"""
    try:
        sol = model.solve(verbosity=0)
    except TypeError:
        sol = model.localsolve(verbosity=0)

    if guess == "ones":
        nvars = len(sol['freevariables'])
        if logspace:
            x0string = ["x0 = zeros({0},1);\n".format(nvars)]
        else:
            x0string = ["x0 = ones({0},1);\n".format(nvars)]
    else:
        x0string = ["x0 = ["]
        i = 1
        for vk in newlist:
            xf = mag(sol['freevariables'][vk])
            if guess == "almost-exact-solution":
                x0 = round(xf, -int(floor(log10(abs(xf))))) # rounds to 1sf
            elif guess == "order-of-magnitude-floor":
                x0 = 10**floor(log10(xf))
            elif guess == "order-of-magnitude-round":
                x0 = 10**round(log10(xf))
            elif isinstance(guess, list):
                x0 = guess[i-1]
            else:
                raise Exception("Unexpected guess type")

            if logspace:
                x0 = log(x0)
            x0string += [str(x0) + ", "]
            i += 1
        x0string += ["];\n"]

    return "".join(x0string)
