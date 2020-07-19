"from gpkit/tests/t_vars.py"

def test_shapes(self):
    with gpkit.Vectorize(3):
        with gpkit.Vectorize(5):
            y = gpkit.Variable("y")
            x = gpkit.VectorVariable(2, "x")
        z = gpkit.VectorVariable(7, "z")

    self.assertEqual(y.shape, (5, 3))
    self.assertEqual(x.shape, (2, 5, 3))
    self.assertEqual(z.shape, (7, 3))