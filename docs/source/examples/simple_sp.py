"""Adapted from t_SP in tests/t_geometric_program.py"""
from gpkit import Model, Variable, SignomialsEnabled

# Decision variables
x = Variable('x')
y = Variable('y')

# must enable signomials for subtraction
with SignomialsEnabled():
    constraints = [x >= 1-y, y <= 0.1]

# create and solve the SP
m = Model(x, constraints)
print(m.localsolve(verbosity=0).summary())
assert abs(m.solution(x) - 0.9) < 1e-6

# full interim solutions are available
print("x values of each GP solve (note convergence)")
print(", ".join("%.5f" % sol["freevariables"][x] for sol in m.program.results))
