import numpy as np
from gpkit import *

x = Variable("x", choices=range(1,4))
num = Variable("numerator", np.linspace(0.5, 7, 11))

m = Model(x + num/x)
sol = m.solve(verbosity=0)

print(sol.table())
