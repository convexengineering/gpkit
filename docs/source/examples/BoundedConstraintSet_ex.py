from gpkit import Variable, Model
from gpkit.tools import BoundedConstraintSet

x = Variable("x")
m = Model(1/x, BoundedConstraintSet([x >= 1]))
m.solve()
