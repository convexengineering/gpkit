from gpkit import Variable, Model, settings
from gpkit.constraints.bounded import BoundedConstraintSet

A = Variable("A")
D = Variable("D")
F = Variable("F")
mi = Variable("m_i")

Fs = Variable("Fs", 0.9)
mb = Variable("m_b", 0.4)
V = Variable("V", 300)

constraints = [F >= D + V**2,
               D == V**2*A,
               V >= mi + mb,
               Fs <= mi,
               ]

# Model(F, constraints).solve("mosek")  # does not solve
if "mosek" in settings["installed_solvers"]:
    m = Model(F, BoundedConstraintSet(constraints))
    # by default, prints bounds warning during solve
    sol = m.solve("mosek", verbosity=0)
    # bound waring is also accessible in sol:
    print sol["boundedness"]
