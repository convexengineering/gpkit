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
        latexleft = trycall(self.left, "latex", excluded, str(self.left))
        latexright = trycall(self.right, "latex", excluded, str(self.right))
        return ("%s %s %s" % (latexleft, latex_oper, latexright))

    def sub(self, subs):
        "Returns a substituted version of this constraint."
        subbedleft = trycall(self.left, "sub", subs, self.left)
        subbedright = trycall(self.right, "sub", subs, self.right)
        subbed = self.func_opers[self.oper](subbedleft, subbedright)
        subbed.substitutions = self.substitutions
        return subbed

    def process_result(self, result):
        "Process solver results"
        pass


def trycall(obj, attr, arg, default):
    "Try to call method of an object, returning `default` if it does not exist"
    if hasattr(obj, attr):
        return getattr(obj, attr)(arg)
    return default
