 from gpkit import Variable, Model
 from gpkit.tools import BoundedConstraintSet
 
 x = Variable("x")
 m = Model(x, BoundedConstraintSet([x <= 1]))
 m.solve()
