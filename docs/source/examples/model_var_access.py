"Demo of accessing variables in models"
from gpkit import Model, Variable


class Battery(Model):
    """A simple battery

    Upper Unbounded
    ---------------
    m

    Lower Unbounded
    ---------------
    E

    """
    def setup(self):
        h = Variable("h", 200, "Wh/kg", "specific energy")
        E = self.E = Variable("E", "MJ", "stored energy")
        m = self.m = Variable("m", "lb", "battery mass")
        return [E <= m*h]


class Motor(Model):
    """Electric motor

    Upper Unbounded
    ---------------
    m

    Lower Unbounded
    ---------------
    Pmax

    """
    def setup(self):
        m = self.m = Variable("m", "lb", "motor mass")
        f = Variable("f", 20, "lb/hp", "mass per unit power")
        Pmax = self.Pmax = Variable("P_{max}", "hp", "max output power")
        return [m >= f*Pmax]


class PowerSystem(Model):
    """A battery powering a motor

    Upper Unbounded
    ---------------
    m

    Lower Unbounded
    ---------------
    E, Pmax

    """
    def setup(self):
        battery, motor = Battery(), Motor()
        components = [battery, motor]
        m = self.m = Variable("m", "lb", "mass")
        self.E = battery.E
        self.Pmax = motor.Pmax

        return [components,
                m >= sum(comp.m for comp in components)]

PS = PowerSystem()
print "Getting the only var 'E': ", PS["E"]
print "The top-level var 'm': ", PS.m
print "All the variables 'm': ", PS.variables_byname("m")
