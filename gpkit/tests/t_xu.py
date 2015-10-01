from gpkit import SignomialsEnabled, Variable, Model

def test_xu_ineq():
    x = Variable('x')
    y = Variable('y')
    with SignomialsEnabled():
         m  = Model(x, [x >= 0.1,
                        x + y >= 1,
                        x + y <= 1])
    sol = m.localsolve(algorithm="Xu")

def test_xu_eq():
    x = Variable('x')
    y = Variable('y')
    with SignomialsEnabled():
        m  = Model(x, [x >= 0.1,
                       x + y == 1])
    sol = m.localsolve(algorithm="Xu")

test_xu_ineq()
test_xu_eq()
