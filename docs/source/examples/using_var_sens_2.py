"Example variable sensitivity usage"
import math
import gpkit
x = gpkit.Variable("x")
x_squared_min = gpkit.Variable("x^2_{min}", 2)
sol = gpkit.Model(x, [x_squared_min <= x**2]).solve()
sens_x_min = sol["sensitivities"]["variables"][x_squared_min]
assert math.isclose(sens_x_min, 0.50, rel_tol=0.01)
