from cvxopt import matrix, spmatrix, solvers, log, exp


def demo_unexpected_dual_sol():
    '''solve the gp
    minimize    x + 2y
    subject to  1 > 1/(xy)
    '''
    K = [2, 1]
    F = spmatrix([1, 1, -1, -1], [0, 1, 2, 2], [0, 1, 0, 1], tc='d')
    g = log(matrix([1, 2, 1]))
    sol = solvers.gp(K, F, g)
    print 'primal solution is:   (x=%s, y=%s)' % tuple(exp(sol['x']))
    print '  dual solution is:  %s' % sol['y'].__repr__()


if __name__ == '__main__':
    demo_unexpected_dual_sol()
