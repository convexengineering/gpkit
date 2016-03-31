from gpkit import SignomialsEnabled
from test import test

def fmincon(m):
    i = 1
    newdict = {}
    for key in m.varkeys:
        newdict[key] = 'x({0})'.format(i)
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
            fmccon += [cc.str_without()]

    with open('confuntest.m', 'w') as outfile:
        outfile.write(
                       "function [c, ceq] = confun(x)\n" + 
                       "% Nonlinear inequality constraints\n" + 
                       "c = [\n" + 
                       "\n".join(fmccon) + 
                       "    ];\n"
                      )
    return fmccon

if __name__ == '__main__':
    m = test()
    fmccon = fmincon(m)
