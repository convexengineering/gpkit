"verifies that bounds are caught through monomials"
from gpkit import Model, parse_variables


class BoundsChecking(Model):
    """Implements a crazy set of unbounded variables.

    Variables
    ---------
    Ap          [-]  d
    D           [-]  e
    F           [-]  s
    mi          [-]  c
    mf          [-]  r
    T           [-]  i
    nu          [-]  p
    Fs    0.9   [-]  t
    mb    0.4   [-]  i
    rf    0.01  [-]  o
    V   300     [-]  n

    Upper Unbounded
    ---------------
    F

    Lower Unbounded
    ---------------
    D

    """
    def setup(self):
        exec parse_variables(BoundsChecking.__doc__)
        self.cost = F
        return [
            F >= D + T,
            D == rf*V**2*Ap,
            Ap == nu,
            T == mf*V,
            mf >= mi + mb,
            mf == rf*V,
            Fs <= mi
        ]


m = BoundsChecking()
print m.str_without(["models"])
try:
    m.solve()
except ValueError:
    pass
gp = m.gp(allow_missingbounds=True)

bplate = ", but would gain it from any of these sets of bounds: "
assert {(m.D.key, 'lower'): bplate + "[(%s, 'lower')]" % m.Ap,
        (m.Ap.key, 'lower'): bplate + ("[(%s, 'lower')]"
                                       " or [(%s, 'lower')]" % (m.D, m.nu)),
        (m.nu.key, 'lower'): bplate + "[(%s, 'lower')]" % m.Ap
       } == gp.missingbounds
