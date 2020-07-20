from gpkit import Variable, Model
from gpkit.constraints.loose import Loose

Loose.reltol = 1e-4  # set the global tolerance of Tight
x = Variable('x')
x_min = Variable('x_{min}', 1)
m = Model(x, [Loose([x >= 2], senstol=1e-4),  # set the specific tolerance
                x >= x_min])
m.solve(verbosity=0)  # prints warning