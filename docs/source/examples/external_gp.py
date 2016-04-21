"Can be found in gpkit/docs/source/examples/external_gp.py"

import numpy as np
from gpkit import Variable, Model
from external_class import External_Constraint

x = Variable("x")
y = Variable("y")

objective = y

constraints = [External_Constraint(x, y),
               x <= np.pi/2.,
               x >= np.pi/4.,
              ]

m = Model(objective, constraints)
sol = m.localsolve(verbosity=1)
