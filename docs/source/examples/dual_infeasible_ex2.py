from gpkit import Variable, Model
x = Variable("x")
y = Variable("y")
m = Model(x**0.01 * y, [x*y >= 1])

#rasises RuntimeWarning, unknown on cvxopt DUAL_INFEASIBLE_CER on Mosek
#m.solve()
