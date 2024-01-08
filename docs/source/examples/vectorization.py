"Example Vectorize usage, from gpkit/tests/t_vars.py"
from gpkit import Variable, Vectorize, VectorVariable

with Vectorize(3):
    with Vectorize(5):
        y = Variable("y")
        x = VectorVariable(2, "x")
    z = VectorVariable(7, "z")

assert y.shape == (5, 3)
assert x.shape == (2, 5, 3)
assert z.shape == (7, 3)
