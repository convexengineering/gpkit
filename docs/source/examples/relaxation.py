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
from gpkit.constraints.relax import RelaxAll
relaxall = RelaxAll(m)
print relaxall
print relaxall.solve(verbosity=0).table()  # solves with an x of 1.414
print

print "With constraints relaxed individually"
print "====================================="
from gpkit.constraints.relax import RelaxConstraints
relaxconstraints = RelaxConstraints(m)
relaxconstraints.cost *= m.cost**0.01  # add a bit of the original cost in
print relaxconstraints
print relaxconstraints.solve(verbosity=0).table()  # solves with an x of 1.0
print

print "With constants relaxed individually"
print "==================================="
from gpkit.constraints.relax import RelaxConstants
relaxedconstants = RelaxConstants(m)
relaxedconstants.cost *= m.cost**0.01  # add a bit of the original cost in
print relaxedconstants
print relaxedconstants.solve(verbosity=0).table()  # brings x_min down to 1.0
print
