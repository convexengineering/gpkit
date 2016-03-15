"Implements SingleEquationConstraint"
from operator import le, ge, eq
from ..small_scripts import try_str_without
from ..repr_conventions import _str, _repr, _repr_latex_


class SingleEquationConstraint(object):
    "Constraint expressible in a single equation."

    latex_opers = {"<=": "\\leq", ">=": "\\geq", "=": "="}
    func_opers = {"<=": le, ">=": ge, "=": eq}

    __str__ = _str
    __repr__ = _repr
    _repr_latex_ = _repr_latex_

    def __init__(self, left, oper, right):
        self.left = left
        self.oper = oper
        self.right = right
        self.substitutions = {}

    def str_without(self, excluded=None):
        "String representation without attributes in excluded list"
        if excluded is None:
            excluded = ["units"]
        return "%s %s %s" % (try_str_without(self.left, excluded),
                             self.oper,
                             try_str_without(self.right, excluded))

    def subconstr_str(self, excluded):
        "The collapsed string of a constraint"
        pass

    def subconstr_latex(self, excluded):
        "The collapsed latex of a constraint"
        pass

    def latex(self, excluded=None):
        "Latex representation without attributes in excluded list"
        if not excluded:
            excluded = ["units"]  # previously bool(self.left.units)
        latex_oper = self.latex_opers[self.oper]
        return ("%s %s %s" % (self.left.latex(excluded), latex_oper,
                              self.right.latex(excluded)))

    def sub(self, subs, value=None):
        "Returns a substituted version of this constraint."
        subbed = self.func_opers[self.oper](self.left.sub(subs, value),
                                            self.right.sub(subs, value))
        subbed.substitutions = self.substitutions
        return subbed

    def process_result(self, result):
        "Process solver results"
        pass
