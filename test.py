from gpkit import *
from gpkit.nomials.array import NomialArray
import numpy as np

t = Variable("t")
u = Variable("u")
v = Variable("v")
w = Variable("w")
x = VectorVariable(3, "x")
y = VectorVariable(3, "y")
z = VectorVariable(3, "z")
a = VectorVariable((3, 2), "a")
b = VectorVariable((3, 2), "b")

nni = 3
ii = np.arange(1., nni+1.)
ii = np.tile(ii, a.shape[1:]+(1,)).T
print NomialArray(ii)#/nni
print a == w*ii/nni
# print w >= (x[0]*t + x[1]*u)/v
# print w >= x
# print w*np.array([1, 2, 3])
assert str(x) == "x[:]"
assert str(x*2) == "x[:]*2"
assert str(2*x) == "2*x[:]"
assert str(x + 2) == "x[:] + 2"
assert str(2 + x) == "2 + x[:]"
assert str(x/2) == "x[:]/2"
assert str(2/x) == "2/x[:]"
assert str(x**3) == "x[:]^3"
assert str(-x) == "-x[:]"
assert str(x/y/z) == "x[:]/y[:]/z[:]"
assert str(x/(y/z)) == "x[:]/(y[:]/z[:])"
assert str(x >= y) == "x[:] >= y[:]"
assert str(x >= y + z) == "x[:] >= y[:] + z[:]"
assert str(x[:2]) == "x[:2]"
assert str(x[:]) == "x[:]"
assert str(x[1:]) == "x[1:]"
assert str(y * [1, 2, 3]) == "y[:]*[1, 2, 3]"
assert str(x[:2] == (y*[1, 2, 3])[:2]) == "x[:2] = (y[:]*[1, 2, 3])[:2]"
assert str(y + [1, 2, 3]) == "y[:] + [1, 2, 3]"
# TODO: print x == y + [1, 2, 3] should raise a clearer error
assert str(x >= y + [1, 2, 3]) == "x[:] >= y[:] + [1, 2, 3]"
assert a[:, 0] == "a[:,0]"
assert a[2, :] == "a[2,:]"
g = a[2, 0]
gstrbefore = str(g)
g.ast = None
gstrafter = str(g)
assert gstrbefore == gstrafter
