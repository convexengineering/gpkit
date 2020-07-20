from gpkit import Variable

# create a Posynomial expression x + xy^2
x = Variable("x")
y = Variable("y")
p = x + x * y**2
print(type(p))  # gpkit.nomials.Posynomial