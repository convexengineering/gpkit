"Debug examples"

from gpkit import Variable, Model, units

x = Variable("x", "ft")
x_min = Variable("x_min", 2, "ft")
x_max = Variable("x_max", 1, "ft")
y = Variable("y", "volts")

m = Model(x/y, [x <= x_max, x >= x_min])
m.debug()

print "# Now let's try a model unsolvable with relaxed constants\n"

Model(x, [x <= units("inch"), x >= units("yard")]).debug()

print "# And one that's only unbounded\n"

# the value of x_min was used up in the previous model!
x_min = Variable("x_min", 2, "ft")
Model(x/y, [x >= x_min]).debug()
