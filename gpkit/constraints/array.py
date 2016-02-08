from .set import ConstraintSet
from .single_equation import SingleEquationConstraint


class ArrayConstraint(SingleEquationConstraint, ConstraintSet):
    left = None
    oper = None
    right = None
    substitutions = None

    def __new__(cls, input_array, left, oper, right):
        "Constructor. Required for objects inheriting from np.ndarray."
        obj = ConstraintSet.__new__(cls, input_array, {})
        obj.left = left
        obj.oper = oper
        obj.right = right
        return obj

    def __array_finalize__(self, obj):
        "Finalizer. Required for objects inheriting from np.ndarray."
        if obj is None:
            return
        ConstraintSet.__array_finalize__(self, obj)

    def str_without(self, excluded=["units"]):
        if self.oper:
            return SingleEquationConstraint.str_without(self, excluded)
        else:
            return NomialArray.str_without(self, excluded)

    def latex(self, *args, **kwargs):
        if self.oper:
            return SingleEquationConstraint.latex(self)
        else:
            return NomialArray.latex(self, *args, **kwargs)
