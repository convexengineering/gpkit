from gpkit import Variable, Model
from gpkit.tools import determine_unbounded_variables

A = Variable("A")
D = Variable("D")
F = Variable("F")
mi = Variable("m_i")

Fs = Variable("Fs", 0.9)
mb = Variable("m_b", 0.4)
V = Variable("V", 300)

m = Model(F, [F >= D + V**2,
              D == V**2*A,
              V >= mi + mb,
              Fs <= mi,
             ])
# sol = m.solve("mosek")  # returns UNKNOWN
bounds = determine_unbounded_variables(m, "mosek")
print bounds
