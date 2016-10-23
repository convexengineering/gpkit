from gpkit import Variable, Model

#Make the necessary Variables
x = Variable("x")
y = Variable("y")

#make the constraints
constraints = [
    x >= 1,
    0.5 <= x*y,
    x*y <= 1.5
    ]

#substitute a value for y
substitutions = {
    "y": 2
    }

#declare the objective
objective = x*y

#construct the model
m = Model(objective, constraints, substitutions)

#solve the model
#raises RuntimeWarning uknown on cvxopt and RuntimeWarning
#PRIM_INFES_CER with mosek
#m.solve()
