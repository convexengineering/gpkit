"Example Vectorize usage, from gpkit/tests/t_vars.py"
import gpkit

with gpkit.Vectorize(3):
    with gpkit.Vectorize(5):
        y = gpkit.Variable("y")
        x = gpkit.VectorVariable(2, "x")
    z = gpkit.VectorVariable(7, "z")

assert(y.shape == (5, 3))
assert(x.shape == (2, 5, 3))
assert(z.shape == (7, 3))
