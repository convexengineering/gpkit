from .set import ConstraintSet
from .single_equation import SingleEquationConstraint
from ..nomials.array import NomialArray


class ArrayConstraint(ConstraintSet, SingleEquationConstraint):
    def __init__(self, constraints, left, oper, right):
        ConstraintSet.__init__(self, constraints)
        self.left = left
        self.oper = oper
        self.right = right
