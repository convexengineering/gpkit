from gpkit import Variable, Model
x = Variable("x")
m = Model(x, [1/x >= 1])

#rasises RuntimeWarning, unknown on cvxopt DUAL_INFEASIBLE_CER on Mosek
#m.solve()
