"Implements SingleEquationConstraint"
from operator import le, ge, eq
from ..small_scripts import try_str_without


class SingleEquationConstraint(object):
    "Constraint expressible in a single equation."
    latex_opers = {"<=": "\\leq", ">=": "\\geq", "=": "="}
    func_opers = {"<=": le, ">=": ge, "=": eq}

    def __repr__(self):
        return "gpkit.%s(%s)" % (self.__class__.__name__, self)

    def __str__(self):
        return self.str_without(["units"])

    def str_without(self, excluded=[]):
        return "%s %s %s" % (try_str_without(self.left, excluded),
                             self.oper,
                             try_str_without(self.right, excluded))

    def latex(self):
        latex_oper = self.latex_opers[self.oper]
        showunits = False  # previously bool(self.left.units)
        return ("%s %s %s" % (self.left.latex(showunits=showunits), latex_oper,
                              self.right.latex(showunits=showunits)))

    def sub(self, subs, value=None):
        "Returns a substituted version of this constraint."
        if value:
            subs = {subs: value}
        subbed = self.func_opers[self.oper](self.left.sub(subs),
                                            self.right.sub(subs))
        subbed.substitutions = self.substitutions
        return subbed

    def process_result(self, result):
        "Process solver results"
        pass
