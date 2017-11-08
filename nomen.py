from gpkit import *


class Cube(Model):
    """Demonstration of nomenclature syntax

    Lines that end in "Variables" will be parsed as a scalar variable table
    until the next blank line.

    Variables
    ---------
    A       [m^2]  surface area
    V   100 [L]    minimum volume

    Lines that end in "Variables of length $N" will be parsed as vector
    variables of length $N until the next blank line.

    Variables of length 3
    ---------------------
    s       [m]    side length

    The above variables are sufficient, but let's introduce more anyway:

    Other Variables
    ---------------
    h     1 [m]    minimum height

    Upper Unbounded
    ---------------
    A, V

    """
    def setup(self):
        print parse_variables(Cube.__doc__)
        exec parse_variables(Cube.__doc__)

        return [A >= 2*(s[0]*s[1] + s[1]*s[2] + s[2]*s[0]),
                s.prod() >= V,
                s[2] >= h]


c = Cube()
c.cost = c.A
print c.solve().table()

verify_model(Cube)
