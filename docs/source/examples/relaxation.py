"Relaxation examples"

from gpkit import Variable, Model
x = Variable("x")
x_min = Variable("x_min", 2)
x_max = Variable("x_max", 1)
m = Model(x, [x <= x_max, x >= x_min])
print "Original model"
print "=============="
print m
print
# m.solve()  # raises a RuntimeWarning!

print "With constraints relaxed equally"
print "================================"
from gpkit.constraints.relax import ConstraintsRelaxedEqually
allrelaxed = ConstraintsRelaxedEqually(m)
mr1 = Model(allrelaxed.relaxvar, allrelaxed)
print mr1
print mr1.solve(verbosity=0).table()  # solves with an x of 1.414
print

print "With constraints relaxed individually"
print "====================================="
from gpkit.constraints.relax import ConstraintsRelaxed
constraintsrelaxed = ConstraintsRelaxed(m)
mr2 = Model(constraintsrelaxed.relaxvars.prod() * m.cost**0.01,
            # add a bit of the original cost in
            constraintsrelaxed)
print mr2
print mr2.solve(verbosity=0).table()  # solves with an x of 1.0
print

print "With constants relaxed individually"
print "==================================="
from gpkit.constraints.relax import ConstantsRelaxed
constantsrelaxed = ConstantsRelaxed(m)
mr3 = Model(constantsrelaxed.relaxvars.prod() * m.cost**0.01,
            # add a bit of the original cost in
            constantsrelaxed)
print mr3
print mr3.solve(verbosity=0).table()  # brings x_min down to 1.0
print
