"Example post-solve evaluated variable"
from gpkit import Variable, Model

# code from t_constraints.test_evalfn in tests/t_sub.py
x = Variable("x")
x2 = Variable("x^2", evalfn=lambda v: v[x]**2)
m = Model(x, [x >= 2])
m.unique_varkeys = set([x2.key])
sol = m.solve(verbosity=0)
assert abs(sol(x2) - 4) <= 1e-4
