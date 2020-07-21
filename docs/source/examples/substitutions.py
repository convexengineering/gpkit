"Example substitution; adapted from t_sub.py/t_NomialSubs /test_Basic"
from gpkit import Variable
x = Variable("x")
p = x**2
assert p.sub({x: 3}) == 9
assert p.sub({x.key: 3}) == 9
assert p.sub({"x": 3}) == 9
