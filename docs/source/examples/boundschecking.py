"verifies that bounds are caught through monomials"
from gpkit import Variable, Model

Ap = Variable('A_p')
D = Variable('D')
F = Variable('F')
mi = Variable('m_i')
mf = Variable('m_f')
T = Variable('T')

Fs = Variable('Fs', 0.9)
mb = Variable('m_b', 0.4)
rf = Variable('r_f', 0.01)
V = Variable('V', 300)
nu = Variable("\\nu")

m = Model(F,
          [F >= D + T,
           D == rf*V**2*Ap,
           Ap == nu,
           T == mf*V,
           mf >= mi + mb,
           mf == rf*V,
           Fs <= mi,
          ])
print m
try:
    m.solve()
except ValueError:
    pass
gp = m.gp(allow_missingbounds=True)

bplate = ", but would gain it from any of these sets of bounds: "
assert {(D.key, 'lower'): bplate + "[(A_p, 'lower')]",
        (Ap.key, 'lower'): bplate + "[(D, 'lower')] or [(\\nu, 'lower')]",
        (nu.key, 'lower'): bplate + "[(A_p, 'lower')]"
       } == gp.missingbounds
