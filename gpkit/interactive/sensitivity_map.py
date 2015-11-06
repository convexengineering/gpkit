import numpy as np
from gpkit import Monomial
from gpkit.small_scripts import mag, latex_num

BLUE = np.array([16,131,246])/255.0
RED = np.array([211,24,16])/255.0
GRAY = 0.7*np.ones(3)

def colorfn_gen(scale, power=0.66):
    scale = float(scale)
    def colorfn(senss):
        if senss < 0:
            senss = -senss
            color = BLUE
        else:
            color = RED
        blended_color = GRAY + (color-GRAY)*(senss/scale)**power
        return "[rgb]{%.2f,%.2f,%.2f}"% tuple(blended_color)
    return colorfn

def sig_senss_latex(sig, sol, colorfn):
    "For pretty printing with Sympy"
    mstrs = []
    for c, exp in zip(sig.cs, sig.exps):
        pos_vars, neg_vars = [], []
        for var, x in exp.items():
            varlatex = var._latex()
            # painting
            senss = sol["sensitivities"]["variables"][var]
            colorstr = colorfn(senss)
            varlatex = "\\textcolor%s{%s}" % (colorstr, varlatex)
            # end painting
            if x > 0:
                pos_vars.append((varlatex, x))
            elif x < 0:
                neg_vars.append((varlatex, x))

        pvarstrs = ['%s^{%.2g}' % (varl, x) if "%.2g" % x != "1" else varl
                    for (varl, x) in pos_vars]
        nvarstrs = ['%s^{%.2g}' % (varl, -x)
                    if "%.2g" % -x != "1" else varl
                    for (varl, x) in neg_vars]
        pvarstrs.sort()
        nvarstrs.sort()
        pvarstr = ' '.join(pvarstrs)
        nvarstr = ' '.join(nvarstrs)
        c = mag(c)
        cstr = "%.2g" % c
        if pos_vars and (cstr == "1" or cstr == "-1"):
            cstr = cstr[:-1]
        else:
            cstr = latex_num(c)

        if not pos_vars and not neg_vars:
            mstr = "%s" % cstr
        elif pos_vars and not neg_vars:
            mstr = "%s%s" % (cstr, pvarstr)
        elif neg_vars and not pos_vars:
            mstr = "\\frac{%s}{%s}" % (cstr, nvarstr)
        elif pos_vars and neg_vars:
            mstr = "%s\\frac{%s}{%s}" % (cstr, pvarstr, nvarstr)

        mstrs.append(mstr)

    return " + ".join(sorted(mstrs))


class SensitivityMap(object):

    def __init__(self, model):
        self.model = model
        self.costlatex = model.cost._latex()
        self.constraints = model.constraints

    @property
    def solution(self):
        if not hasattr(self.model, "solution"):
            self.model.solve()
        if len(self.model.solution) > 1:
            return self.model.solution.atindex(0)  # TODO: support sweeps
        else:
            return self.model.solution

    @property
    def constraint_latex_list(self):
        return ["    & %s \\\\" % self.constraint_latex(constr)
                for constr in self.constraints]

    def constraint_latex(self, constraint):
        left = self.signomial_latex(constraint.left)
        right = self.signomial_latex(constraint.right)
        return left + constraint.oper_l + right

    def signomial_latex(self, signomial):
        maxsenss = max(self.solution["sensitivities"]["variables"].values())
        colorfn = colorfn_gen(maxsenss, 0.66)
        return sig_senss_latex(signomial, self.solution, colorfn)

    @property
    def latex(self):
        return "\n".join(["\\color[gray]{%.2f}" % GRAY[0],
                          "\\begin{array}[ll]",
                          "\\text{}",
                          "\\text{minimize}",
                          "    & %s \\\\" % self.costlatex,
                          "\\text{subject to}"] +
                         self.constraint_latex_list +
                         ["\\end{array}"])

    def _repr_latex_(self):
        return "$$\\require{color}\n"+self.latex+"\n$$"
