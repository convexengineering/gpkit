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

print w >= x
