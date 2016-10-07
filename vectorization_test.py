import gpkit

from gpkit.nomials.variables import VectorizableVariable as VV

with gpkit.vectorize(3):
    with gpkit.vectorize(5):
        x = VV("x")

m = gpkit.Model(x.prod(), [x >= 1])
print m.solve().table()

with gpkit.vectorize(3):
    with gpkit.vectorize(5):
        y = gpkit.Variable("y")
        x = gpkit.VectorVariable(2, "x")
    z = gpkit.VectorVariable(7, "z")


assert y.shape == (5,3)
assert x.shape == (2,5,3)
assert z.shape == (7,3)
