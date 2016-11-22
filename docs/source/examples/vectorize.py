from gpkit import Model, Variable, Vectorize

class Test(Model):
	def __init__(self):
		x = Variable("x")
		Model.__init__(self, None, [x >= 1])

print "SCALAR"
m = Test()
m.cost = m["x"]
print m.solve(verbosity=0).table()

print "__________\n"
print "VECTORIZED"
with Vectorize(3):
	m = Test()
m.cost = m["x"].prod()
m.append(m["x"][1] >= 2)
print m.solve(verbosity=0).table()