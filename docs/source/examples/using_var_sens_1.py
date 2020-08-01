"Example variable sensitivity usage"
import math
from gpkit import Variable, Model
x = Variable("x")
x_min = Variable("x_{min}", 2)
sol = Model(x, [x_min <= x]).solve()
sens_x_min = sol["sensitivities"]["variables"][x_min]
assert math.isclose(sens_x_min, 1.00, rel_tol=0.01)
