"Can be found in gpkit/docs/source/examples/external_sp.py"
import numpy as np
from gpkit import Variable, Model

x = Variable("x")


def y_ext(self, x0):
    "Returns constraints on y derived from x0"
    if x.key not in x0:
        return self >= x
    else:
        return self >= x/x0[x] * np.sin(x0[x])

y = Variable("y", externalfn=y_ext)

m = Model(y, [np.pi/4 <= x, x <= np.pi/2])
print m.localsolve(verbosity=0).summary()
