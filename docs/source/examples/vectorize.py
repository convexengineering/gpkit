"Vectorization demonstration"
from gpkit import Model, Variable, Vectorize

class Test(Model):
    """A simple scalar model

    Upper Unbounded
    ---------------
    x
    """
    def setup(self):
        x = self.x = Variable("x")
        return [x >= 1]

print "SCALAR"
m = Test()
m.cost = m["x"]
print m.solve(verbosity=0).summary()

print "__________\n"
print "VECTORIZED"
with Vectorize(3):
    m = Test()
m.cost = m["x"].prod()
m.append(m["x"][1] >= 2)
print m.solve(verbosity=0).summary()
