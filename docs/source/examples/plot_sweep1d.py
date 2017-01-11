"Demonstrates manual and auto sweeping with plot_sweep1d"
import numpy as np
from gpkit import Model, Variable, units
from gpkit.interactive import plot_sweep1d

x = Variable("x", "m", "Swept Variable")
y = Variable("y", "m^2", "Cost")
m = Model(y, [y >= (x/2)**-0.5 * units.m**2.5 + 1*units.m**2, y >= (x/2)**2])

# manual sweep
f, ax = plot_sweep1d(m, x, np.linspace(1, 3, 20), y)
ax.set_title("Manually swept, 20 points")
f.show()
f.savefig("plot_sweep1d.png")

# autosweep
f, ax = plot_sweep1d(m, x, (1, 3, 0.001), y)
ax.set_title("Autoswept, 7 points, guaranteed to be in shaded region")
f.show()
f.savefig("plot_autosweep1d.png")
