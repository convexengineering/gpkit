"Another simple primal infeasible example"
from gpkit import Variable, Model

x = Variable("x")
y = Variable("y", 2)

constraints = [
    x >= 1,
    0.5 <= x*y,
    x*y <= 1.5
    ]

objective = x*y
m = Model(objective, constraints)

# raises UnknownInfeasible on cvxopt and PrimalInfeasible on mosek
# m.solve()
