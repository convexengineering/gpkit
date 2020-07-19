from gpkit import Variable, Model
x = Variable("x")
y = Variable("y", 3)  # fix value to 3
m = Model(x, [x >= 1 + y, y >= 1])
_ = m.solve()  # optimal cost is 4; y appears in sol["constants"]

del m.substitutions["y"]
_ = m.solve()  # optimal cost is 2; y appears in Free Variables