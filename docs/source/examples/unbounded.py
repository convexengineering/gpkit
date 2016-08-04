from gpkit import Variable, Model, settings
from gpkit.tools import BoundedConstraintSet

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

# Model(F, constraints).solve("mosek")  # returns UNKNOWN
if "mosek" in settings["installed_solvers"]:
    sol = Model(F, BoundedConstraintSet(constraints)).solve("mosek")
    print sol["boundedness"]
