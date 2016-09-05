"Demonstrate a trivial unbounded variable"
from gpkit import Variable, Model
from gpkit.constraints.bounded import BoundedConstraintSet

x = Variable("x")

constraints = [x >= 1]

# Model(x, constraints).solve()  # does not solve
m = Model(1/x, BoundedConstraintSet(constraints))
# by default, prints bounds warning during solve
sol = m.solve(verbosity=0)
print sol.table()
print "sol['boundedness'] is:", sol["boundedness"]
