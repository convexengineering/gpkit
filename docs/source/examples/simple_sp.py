"""Adapted from t_SP in tests/t_geometric_program.py"""
import gpkit

# Decision variables
x = gpkit.Variable('x')
y = gpkit.Variable('y')

# must enable signomials for subtraction
with gpkit.SignomialsEnabled():
    constraints = [x >= 1-y, y <= 0.1]

# create and solve the SP
m = gpkit.Model(x, constraints)
sol = m.localsolve(verbosity=0)
print sol.table()
assert abs(sol(x) - 0.9) < 1e-6
