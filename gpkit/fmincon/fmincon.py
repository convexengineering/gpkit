from gpkit import SignomialsEnabled
from test import test

def fmincon(m):
    i = 1
    newdict = {}
    newlist = []
    for key in m.varkeys:
        newdict[key] = 'x({0})'.format(i)
        newlist += ['x_{0}: '.format(i) + key.str_without()]
        i += 1

    constraints = m.program.constraints
    constraints.subinplace(newdict)
    fmccon = []

    with SignomialsEnabled():
        for constraint in constraints:
            if constraint.oper == '<=':
                cc = constraint.left - constraint.right
            elif constraint.oper == '>=':
                cc = constraint.right - constraint.left
            elif constraint.oper == '=':
                cc = constraint.right - constraint.left
            fmccon += [cc.str_without("units")]

    with open('confun.m', 'w') as outfile:
        outfile.write("function [c, ceq] = confun(x)\n" +
                      "% Nonlinear inequality constraints\n" +
                      "c = [\n" +
                      "\n".join(fmccon).replace('**', '.^') +
                      "    ];\n\n" +
                      "ceq = [];"
                     )

    obj = m.cost
    obj.subinplace(newdict)
    fmcobj = obj.str_without("units").replace('**', '.^')

    with open('objfun.m', 'w') as outfile:
        outfile.write("function f = objfun(x)\n" +
                      "f = " + fmcobj + ";\n")

    with open('lookup.txt', 'w') as outfile:
        outfile.write("\n".join(newlist))

    with open('main.m', 'w') as outfile:
        outfile.write("x0 = ones({0},1);\n".format(i-1) +
                      #"options = optimoptions(@fmincon,'Algorithm','sqp');\n" +
                      "[x,fval] = ...\n" +
                      "fmincon(@objfun,x0,[],[],[],[],[],[],@confun,options);")

    return fmcobj, fmccon

if __name__ == '__main__':
    m = test()
    fmccon = fmincon(m)
