"Debug examples"

from gpkit import Variable, Model
x = Variable("x", "m")
x_min = Variable("x_min", 2, "ft")
x_max = Variable("x_max", 1, "ft")
y = Variable("y", "volts")
m = Model(x/y, [x <= x_max, x >= x_min])
m.debug(verbosity=1)
