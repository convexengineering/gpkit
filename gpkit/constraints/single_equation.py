"Implements SingleEquationConstraint"
from __future__ import unicode_literals
from operator import le, ge, eq
from ..small_scripts import try_str_without
from ..repr_conventions import GPkitObject


class SingleEquationConstraint(GPkitObject):
    "Constraint expressible in a single equation."
    latex_opers = {"<=": "\\leq", ">=": "\\geq", "=": "="}
    func_opers = {"<=": le, ">=": ge, "=": eq}

    def __init__(self, left, oper, right):
        self.left, self.oper, self.right = left, oper, right

    def str_without(self, excluded=("units")):
        "String representation without attributes in excluded list"
        return "%s %s %s" % (try_str_without(self.left, excluded),
                             self.oper,
                             try_str_without(self.right, excluded))

    def latex(self, excluded=("units")):
        "Latex representation without attributes in excluded list"
        return ("%s %s %s" % (
            try_str_without(self.left, excluded, latex=True),
            self.latex_opers[self.oper],
            try_str_without(self.right, excluded, latex=True)))
