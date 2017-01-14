"Can be found in gpkit/docs/source/examples/sin_approx_example.py"
import numpy as np
from gpkit import Variable, Model


x = Variable("x")
y = Variable("y")

objective = y

constraints = [y >= x,
               x <= np.pi/2.,
               x >= np.pi/4.,
              ]

m = Model(objective, constraints)
print m.solve(verbosity=0).summary()
