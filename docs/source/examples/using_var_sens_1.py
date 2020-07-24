"Example variable sensitivity usage"
import math
import gpkit
x = gpkit.Variable("x")
x_min = gpkit.Variable("x_{min}", 2)
sol = gpkit.Model(x, [x_min <= x]).solve()
sens_x_min = sol["sensitivities"]["variables"][x_min]
assert math.isclose(sens_x_min, 1.00, rel_tol=0.01)
