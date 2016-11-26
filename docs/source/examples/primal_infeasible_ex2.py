"Another simple primal infeasible example"
from gpkit import Variable, Model

#Make the necessary Variables
x = Variable("x")
y = Variable("y", 2)

#make the constraints
constraints = [
    x >= 1,
    0.5 <= x*y,
    x*y <= 1.5
    ]

#declare the objective
objective = x*y

#construct the model
m = Model(objective, constraints)

#solve the model
#raises RuntimeWarning uknown on cvxopt and RuntimeWarning
#PRIM_INFES_CER with mosek
#m.solve()
