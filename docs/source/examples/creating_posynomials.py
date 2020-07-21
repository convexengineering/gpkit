"Example code for posynomial creation"
from gpkit import Variable
from gpkit.nomials import Posynomial

# create a Posynomial expression x + xy^2
x = Variable("x")
y = Variable("y")
p = x + x * y**2
assert isinstance(p, Posynomial)
