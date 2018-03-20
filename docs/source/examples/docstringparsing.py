"Docstring parsing example"
from gpkit import Model, parse_variables


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

    Let's introduce more variables: (any line ending in "Variables" is parsed)

    Zoning Variables
    ----------------
    h     1 [m]    minimum height

    Upper Unbounded
    ---------------
    A

    The ordering of these blocks doesn't affect anything; order them in the
    way that makes the most sense to someone else reading your model.
    """
    def setup(self):
        exec parse_variables(Cube.__doc__)

        return [A >= 2*(s[0]*s[1] + s[1]*s[2] + s[2]*s[0]),
                s.prod() >= V,
                s[2] >= h]


print parse_variables(Cube.__doc__)
c = Cube()
c.cost = c.A
print c.solve(verbosity=0).table()
