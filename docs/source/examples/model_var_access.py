"Demo of accessing variables in models"
from gpkit import Model, Variable


class Battery(Model):
    "A simple battery"
    def setup(self):
        h = Variable("h", 200, "Wh/kg", "specific energy")
        E = Variable("E", "MJ", "stored energy")
        m = Variable("m", "lb", "battery mass")
        return [E <= m*h]


class Motor(Model):
    "Electric motor"
    def setup(self):
        m = Variable("m", "lb", "motor mass")
        f = Variable("f", 20, "lb/hp", "mass per unit power")
        Pmax = Variable("P_{max}", "hp", "max output power")
        return [m >= f*Pmax]


class PowerSystem(Model):
    "A battery powering a motor"
    def setup(self):
        components = [Battery(), Motor()]
        m = Variable("m", "lb", "mass")
        return [components,
                m >= sum(comp.topvar("m") for comp in components)]

PS = PowerSystem()
print("Getting the only var 'E': ", PS["E"])
print("The top-level var 'm': ", PS.topvar("m"))
print("All the variables 'm': ", PS.variables_byname("m"))
