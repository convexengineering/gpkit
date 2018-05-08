"Show autosweep_1d functionality"
import cPickle as pickle
import numpy as np
import gpkit
from gpkit import units, Variable, Model
from gpkit.tools.autosweep import autosweep_1d
from gpkit.small_scripts import mag

A = Variable("A", "m**2")
l = Variable("l", "m")

m1 = Model(A**2, [A >= l**2 + units.m**2])
tol1 = 1e-3
bst1 = autosweep_1d(m1, tol1, l, [1, 10], verbosity=0)
print "Solved after %2i passes, cost logtol +/-%.3g" % (bst1.nsols, bst1.tol)
# autosweep solution accessing
l_vals = np.linspace(1, 10, 10)
sol1 = bst1.sample_at(l_vals)
print "values of l:", l_vals
print "values of A:", sol1("A")
cost_estimate = sol1["cost"]
cost_lb, cost_ub = sol1.cost_lb(), sol1.cost_ub()
print "cost lower bound:", cost_lb
print "cost estimate:   ", cost_estimate
print "cost upper bound:", cost_ub
# you can evaluate arbitrary posynomials
np.testing.assert_allclose(mag(2*sol1(A)), mag(sol1(2*A)))
assert (sol1["cost"] == sol1(A**2)).all()
# the cost estimate is the logspace mean of its upper and lower bounds
np.testing.assert_allclose((np.log(mag(cost_lb)) + np.log(mag(cost_ub)))/2,
                           np.log(mag(cost_estimate)))
# save autosweep to a file and retrieve it
bst1.save("autosweep.p")
bst1_loaded = pickle.load(open("autosweep.p"))

# this problem is two intersecting lines in logspace
m2 = Model(A**2, [A >= (l/3)**2, A >= (l/3)**0.5 * units.m**1.5])
tol2 = {"mosek": 1e-12, "cvxopt": 1e-7,
        "mosek_cli": 1e-6}[gpkit.settings["default_solver"]]
bst2 = autosweep_1d(m2, tol2, l, [1, 10], verbosity=0)
print "Solved after %2i passes, cost logtol +/-%.3g" % (bst2.nsols, bst2.tol)
print "Table of solutions used in the autosweep:"
print bst2.solarray.table()
