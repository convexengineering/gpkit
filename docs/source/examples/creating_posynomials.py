# create a Posynomial expression x + xy^2
x = Variable("x")
y = Variable("y")
p = x + x * y**2
type(p)  # gpkit.nomials.Posynomial