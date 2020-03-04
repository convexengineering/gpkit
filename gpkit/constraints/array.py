"Implements ArrayConstraint"
from .set import flatiter
from .single_equation import SingleEquationConstraint


# TODO: don't inherit from ConstraintSet, implement own .flat()
# TODO: check for numpy_bools here and here alone
class ArrayConstraint(SingleEquationConstraint, list):
    """A ConstraintSet for prettier array-constraint printing.

    ArrayConstraint gets its `sub` method from ConstrainSet,
    and so `left` and `right` are only used for printing.

    When created by NomialArray `left` and `right` are likely to be
    be either NomialArrays or Varkeys of VectorVariables.
    """
    def __init__(self, constraints, left, oper, right):
        SingleEquationConstraint.__init__(self, left, oper, right)
        list.__init__(self, constraints)

    def lines_without(self, excluded):
        "Returns lines for indentation in hierarchical printing."
        return self.str_without(excluded).split("\n")

    def __bool__(self):
        "Allows the use of '=' NomialArrays as truth elements."
        if self.oper != "=":
            return NotImplemented
        return all(bool(p) for p in flatiter(self))
