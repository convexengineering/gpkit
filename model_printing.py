from gpkit.shortcuts import *
from gpkit import PosyArray

x = Var("x")
x_mod2 = Var("x", model="mod2")
x_mod = Var("x", model="mod")
y = Var("y")
y_mod2 = Var("y", model="mod2")
y_mod = Var("y", model="mod")

xs = PosyArray([x, x_mod, x_mod2])
ys = PosyArray([y, y_mod, y_mod2])

Model(xs.prod(),
      [xs >= ys],
      {ys: [1, 1, 1]}).solve()
