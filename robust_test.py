from gpkit import *

DISTRS = ["normal", "lognormal"]

# x = Variable("x", median=2, sigma=0.5, distr=DISTR, better=-1)
# print Model(x).solve("ecos", ).table()

for DISTR in DISTRS:
    print
    print "********************************"
    print

    x = Variable("x", median=1.5, sigma=0.5, distr=DISTR, better=+1)
    m = Model(1/x)
    m.robust.substitutions["\\Sigma"] = 1
    print m.solve("ecos", verbosity=0).table()

    x = Variable("x", median=1.5, sigma=0.5, distr=DISTR, better=+1)
    y = Variable("y", median=1.5, sigma=0.5, distr=DISTR, better=+1)
    m = Model((x*y)**-0.5)
    m.robust.substitutions["\\Sigma"] = 1
    print m.solve("ecos", verbosity=0).table()

print
print "********************************"
print

# x = Variable("x", median=2, sigma=0.5, distr=DISTR, better=-1)
# y = Variable("y", median=2, sigma=0.5, distr=DISTR, better=-1)
# print Model(x + y).solve("ecos", ).table()
