"Example code for monomial creation"
from gpkit import Variable
from gpkit.nomials import Monomial

# create a Monomial term xy^2/z
x = Variable("x")
y = Variable("y")
z = Variable("z")
m = x * y**2 / z
assert isinstance(m, Monomial)
