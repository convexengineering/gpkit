"Example model formulation"
from gpkit import Variable, Model

x = Variable("x")
y = Variable("y")
z = Variable("z")
S = 200
objective = 1/(x*y*z)
constraints = [2*x*y + 2*x*z + 2*y*z <= S,
               x >= 2*y]
m = Model(objective, constraints)
