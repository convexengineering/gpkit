from gpkit import Variable

# create a Monomial term xy^2/z
x = Variable("x")
y = Variable("y")
z = Variable("z")
m = x * y**2 / z
print(type(m))  # gpkit.nomials.Monomial