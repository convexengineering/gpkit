"Tests non-array linked functions & subs in a vectorization environment"
import numpy as np
from gpkit import Variable, Model, ConstraintSet, Vectorize

class Vehicle(Model):
    "Vehicle model"
    def setup(self):
        self.a = a = Variable("a")
        constraints = [a >= 1]
        return constraints

class System(Model):
    "System model"
    def setup(self):
        with Vectorize(1):
            self.Fleet2 = Fleet2()
        constraints = [self.Fleet2]
        self.cost = sum(self.Fleet2.z)
        return constraints

class Fleet2(Model):
    "Fleet model (composed of multiple Vehicles)"
    def setup(self):
        x = Variable("x")
        lambdafun = lambda c: [c[x]-1, np.ones(x.shape)]
        with Vectorize(2):
            y = Variable("y", lambdafun)
            self.Vehicle = Vehicle()

        self.z = z = Variable("z")
        substitutions = {"x": 4}
        constraints = [
            z >= sum(y/x*self.Vehicle.a),
            self.Vehicle,
        ]
        return constraints, substitutions

m = System()
sol = m.solve()
print(sol.table())

# now with more fleets per system
class System2(Model):
    "System model"
    def setup(self):
        with Vectorize(3):
            self.Fleet2 = Fleet2()
        constraints = [self.Fleet2]
        self.cost = sum(self.Fleet2.z)
        return constraints

m = System2()
sol = m.solve()
print(sol.table())


# now testing substitutions

class Simple(Model):
    "Simple model"
    def setup(self):
        self.x = x = Variable("x")
        y = Variable("y", 1)
        z = Variable("z", 2)
        constraints = [
            x >= y + z,
        ]
        return constraints

class Cake(Model):
    "Cake model"
    def setup(self):
        with Vectorize(3):
            s = Simple()
        c = ConstraintSet([s])
        self.cost = sum(s.x)
        return c

m = Cake()
m.substitutions.update({
    "y": ("sweep", [1, 2, 3]),
    "z": lambda v: v("y")**2,
})
sol = m.solve()
print(sol.table())
