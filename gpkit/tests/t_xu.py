from gpkit import SignomialProgram, SignomialsEnabled, Variable, Model

## Misc code
# m = Model(x, [x >= 0.1,
#               x + y >= 1,
#               x + y <= 1])
#sol = m.localsolve()
#sol = m.localsolve(algorithm="Xu")

def test_xu_ineq():
    x = Variable('x')
    y = Variable('y')
    with SignomialsEnabled():
         sp  = SignomialProgram(x, [x >= 0.1,
                                    x + y >= 1,
                                    x + y <= 1])
    sol = sp.xusolve()
    print sol['variables']['x']
    print sol['variables']['y']

def test_xu_eq():
    x = Variable('x')
    y = Variable('y')
    with SignomialsEnabled():
        sp  = SignomialProgram(x, [x >= 0.1,
                                   x + y == 1])
    #sol = m.localsolve()
    #sol = m.localsolve(algorithm="Xu")
    sol = sp.xusolve()
    print sol['variables']['x']
    print sol['variables']['y']

test_xu_ineq()
test_xu_eq()
