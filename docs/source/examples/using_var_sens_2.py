"Example variable sensitivity usage"
import math
from gpkit import Model, Variable
x = Variable("x")
x_squared_min = Variable("x^2_{min}", 2)
sol = Model(x, [x_squared_min <= x**2]).solve()
sens_x_min = sol["sensitivities"]["variables"][x_squared_min]
assert math.isclose(sens_x_min, 0.50, rel_tol=0.01)
