"""Adapted from t_SP in tests/t_geometric_program.py"""
import gpkit

# Decision variables
x = gpkit.Variable('x')
y = gpkit.Variable('y')

# must enable signomials for subtraction
gpkit.enable_signomials()

# create and solve the SP
sp = gpkit.SP(x, [x >= 1-y, y <= 0.1])
sol = sp.localsolve(printing=False)
assert abs(sol(x) - 0.9) < 1e-6

gpkit.disable_signomials()
