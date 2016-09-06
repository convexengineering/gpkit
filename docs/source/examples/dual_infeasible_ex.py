from gpkit import Variable, Model
x = Variable("x")
m = Model(x, [1/x >= 1])

# m.solve(verbosity=0) would raise a RuntimeWarning
# solver status is unknown on cvxopt and DUAL_INFEASIBLE_CER on Mosek
