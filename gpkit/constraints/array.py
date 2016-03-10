"Implements ArrayConstraint"
from .set import ConstraintSet
from .single_equation import SingleEquationConstraint


class ArrayConstraint(ConstraintSet, SingleEquationConstraint):
    "A ConstraintSet for prettier array-constraint printing"
    def __init__(self, constraints, left, oper, right):
        ConstraintSet.__init__(self, constraints)
        self.left = left
        self.oper = oper
        self.right = right
