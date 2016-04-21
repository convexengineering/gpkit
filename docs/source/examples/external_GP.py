from gpkit import Variable, Model
import numpy as np

x = Variable("x")
y = Variable("y")

objective = y

constraints = [External_Constraint(x),
               x <= np.pi/2.,
               x >= np.pi/4.,
              ]

m = Model(objective, constraints)
sol = m.localsolve()