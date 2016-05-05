"A module to facilitate testing GPkit against fmincon"
from gpkit import SignomialsEnabled
from gpkit.tools.simpleflight import simpleflight
# pylint: disable=redefined-outer-name,invalid-name
# pylint: disable=too-many-statements,too-many-locals

def generate_mfiles(m, writefiles=True):
    """A method for preparing fmincon input files to run a GPkit program"""

    # Create a new dictionary mapping variables to x(i)'s for use w/ fmincon
    i = 1
    newdict = {}
    newlist = []
    original_varkeys = m.varkeys
    for key in m.varkeys:
        if key not in m.substitutions:
            newdict[key] = 'x({0})'.format(i)
            newlist += ['x_{0}: '.format(i) + key.str_without()]
            i += 1

    cost = m.cost # needs to be before subinplace()
    constraints = m
    constraints.subinplace(constraints.substitutions)
    constraints.subinplace(newdict)

    # Make all constraints less than zero, return list of clean strings
    c = [] # inequality constraints
    ceq = [] # equality constraints
    DC = [] # gradients of inequality constraints
    DCeq = [] # gradients of equality constraints
    with SignomialsEnabled():
        for constraint in constraints:
            if constraint.oper == '<=':
                cc = constraint.left - constraint.right
                c += [cc.str_without("units")]
            elif constraint.oper == '>=':
                cc = constraint.right - constraint.left
                c += [cc.str_without("units")]
            elif constraint.oper == '=':
                cc = constraint.right - constraint.left
                ceq += [cc.str_without("units")]

            # Differentiate each constraint w.r.t each variable
            cdm = []
            for key in original_varkeys:
                if key not in m.substitutions:
                    cd = cc.diff(newdict[key])
                    cdm += [cd.str_without("units").replace('**', '.^')]

            if constraint.oper != '=':
                DC += [",...\n          ".join(cdm)]
            else:
                DCeq += [",...\n            ".join(cdm)]

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
                 ";\n          ".join(DC) +
                 "\n         ]';\n    " +
                 "DCeq = [\n            " +
                 ";\n            ".join(DCeq) +
                 "\n           ]';\n" +
                 "end")

    # Differentiate the objective function w.r.t each variable
    objdiff = []
    for key in original_varkeys:
        if key not in m.substitutions:
            costdiff = cost.diff(key)
            costdiff.subinplace(newdict)
            objdiff += [costdiff.str_without("units").replace('**', '.^')]

    # Replace variables with x(i), make clean string using matlab power syntax
    cost.subinplace(newdict)
    obj = cost.str_without("units").replace('**', '.^')

    # String for the objective function .m file
    objfunstr = ("function [f, gradf] = objfun(x)\n" +
                 "f = " + obj + ";\n" +
                 "if nargout > 1\n" +
                 "    gradf  = [" +
                 "\n              ".join(objdiff) +
                 "];\n" +
                 "end")

    # String for main.m
    mainfunstr = ("x0 = ones({0},1);\n".format(i-1) +
                  "options = optimset('fmincon');\n" +
                  "options.Algorithm = 'interior-point';\n" +
                  "options.MaxFunEvals = Inf;\n" +
                  "options.MaxIter = Inf;\n" +
                  "options.GradObj = 'on';\n" +
                  "options.GradConstr = 'on';\n" +
                  "tic;\n" +
                  "[x,fval] = ...\n" +
                  "fmincon(@objfun,x0,[],[],[],[],[],[],@confun,options);\n" +
                  "toc;")

    if writefiles is True:
        # Write the constraint function .m file
        with open('confun.m', 'w') as outfile:
            outfile.write(confunstr)


        # Write the objective function .m file
        with open('objfun.m', 'w') as outfile:
            outfile.write(objfunstr)

        # Write a txt file for looking up original variable names
        with open('lookup.txt', 'w') as outfile:
            outfile.write("\n".join(newlist))

        # Write the main .m file for running fmincon
        with open('main.m', 'w') as outfile:
            outfile.write(mainfunstr)

    return obj, c, ceq, DC, DCeq

if __name__ == '__main__':
    m = simpleflight()
    obj, c, ceq, DC, DCeq = generate_mfiles(m)
