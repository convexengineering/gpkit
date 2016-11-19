"A simple primal infeasible example"
from gpkit import Variable, Model

#Make the necessary Variables
x = Variable("x")
y = Variable("y")

#make the constraints
constraints = [
    x >= 1,
    y >= 2,
    x*y >= 0.5,
    x*y <= 1.5
]

#declare the objective
objective = x*y

#construct the model
m = Model(objective, constraints)

#solve the model
#raises uknown on cvxopt and mosek
#m.solve()
