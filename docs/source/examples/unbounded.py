"Demonstrate a trivial unbounded variable"
from gpkit import Variable, Model, settings
from gpkit.constraints.bounded import BoundedConstraintSet

x = Variable("x")

constraints = [x <= 1]

# Model(x, constraints).solve()  # does not solve
m = Model(x, BoundedConstraintSet(constraints))
# by default, prints bounds warning during solve
sol = m.solve(verbosity=0)
# bound waring is available in sol["boundedness"]:
print sol["boundedness"]
