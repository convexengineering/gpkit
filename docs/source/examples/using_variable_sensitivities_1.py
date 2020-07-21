"Example variable sensitivity usage"
import gpkit
x = gpkit.Variable("x")
x_min = gpkit.Variable("x_{min}", 2)
sol = gpkit.Model(x, [x_min <= x]).solve()
assert sol["sensitivities"]["variables"][x_min] == 1
