"Example of freeing fixed variables"
from gpkit import Variable, Model
x = Variable("x")
y = Variable("y", 3)  # fix value to 3
m = Model(x, [x >= 1 + y, y >= 1])
# verbosity is 0 for testing's sake, no need to do that in your code!
sol = m.solve(verbosity=0)  # optimal cost is 4; y appears in sol["constants"]

assert abs(sol["cost"] - 4) <= 1e-4
assert y in sol["constants"]

del m.substitutions["y"]
sol = m.solve()  # optimal cost is 2; y appears in Free Variables
assert abs(sol["cost"] - 2) <= 1e-4
assert y in sol["freevariables"]
