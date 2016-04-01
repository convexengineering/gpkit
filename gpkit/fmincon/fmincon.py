"A module to facilitate testing GPkit against fmincon"
from gpkit import SignomialsEnabled
from simpleflight import simpleflight

def fmincon(m):
    """A method for preparing fmincon input files to run a GPkit program"""
    i = 1
    newdict = {}
    newlist = []
    for key in m.varkeys:
        if key not in m.substitutions:
            newdict[key] = 'x({0})'.format(i)
            newlist += ['x_{0}: '.format(i) + key.str_without()]
            i += 1

    constraints = m.program.constraints
    constraints.subinplace(constraints.substitutions)
    constraints.subinplace(newdict)
    c = []
    ceq = []

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

    with open('confun.m', 'w') as outfile:
        outfile.write("function [c, ceq] = confun(x)\n" +
                      "% Nonlinear inequality constraints\n" +
                      "c = [\n    " +
                      "\n    ".join(c).replace('**', '.^') +
                      "\n    ];\n\n" +
                      "ceq = [\n      " +
                      "\n      ".join(ceq).replace('**', '.^') + 
                      "\n      ];"
                     )

    cost = m.cost
    cost.subinplace(newdict)
    obj = cost.str_without("units").replace('**', '.^')

    with open('objfun.m', 'w') as outfile:
        outfile.write("function f = objfun(x)\n" +
                      "f = " + obj + ";\n")

    with open('lookup.txt', 'w') as outfile:
        outfile.write("\n".join(newlist))

    with open('main.m', 'w') as outfile:
        outfile.write("x0 = ones({0},1);\n".format(i-1) +
                      #"options = optimoptions(@fmincon,'Algorithm','sqp');\n" +
                      "[x,fval] = ...\n" +
                      "fmincon(@objfun,x0,[],[],[],[],[],[],@confun,options);")

    return obj, c, ceq

if __name__ == '__main__':
    m = simpleflight()
    obj, c, ceq = fmincon(m)
