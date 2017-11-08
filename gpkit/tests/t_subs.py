from gpkit import Variable, Model, SignomialsEnabled

def test_spsubs():

    x = Variable("x", 5)
    y = Variable("y", lambda c: 2*c[x])
    z = Variable("z")
    w = Variable("w")

    with SignomialsEnabled():
        cnstr = [z + w >= y*x, w <= y]

    m = Model(z, cnstr)
    print m.substitutions
    sol = m.localsolve("mosek")
    print m.substitutions

if __name__ == "__main__":
    test_spsubs()



