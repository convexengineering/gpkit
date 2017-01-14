"Demonstrates manual and auto sweeping with plot_sweep1d"
import matplotlib as mpl
mpl.use('Agg')
# comment out the imports above and `show()` to show figures in a window
import numpy as np
from gpkit import Model, Variable, units

x = Variable("x", "m", "Swept Variable")
y = Variable("y", "m^2", "Cost")
m = Model(y, [y >= (x/2)**-0.5 * units.m**2.5 + 1*units.m**2, y >= (x/2)**2])

print "MANUAL SWEEP"
# arguments are: model, swept: values, posnomial for y-axis
sol = m.sweep({x: np.linspace(1, 3, 20)})
f, ax = sol.plot(y)
ax.set_title("Manually swept (20 points)")
# f.show()
f.savefig("plot_sweep1d.png")

print "\nAUTOSWEEP"
# arguments are: model, swept: (min, max, optional logtol), posnomial for y-axis
sol = m.autosweep({x: (1, 3)}, tol=0.001)
f, ax = sol.plot(y)
ax.set_title("Autoswept (7 points)\nGuaranteed to be in blue region")
# f.show()
f.savefig("plot_autosweep1d.png")
