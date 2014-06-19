"""
Example of calling MOSEK expopt
Goal for gpkit/mosek_interface is to make this script work.
"""
def ex1():
    # simplest option -- expose C functionality directly
    # (likely via ctypes)
    # example comes from: 
    #     http://docs.mosek.com/7.0/capi/Exponential_optimization.html
    import numpy as np
    from mosek import Env, Task, prosta, solsta
    from gpkit.mosek_interface import (MSK_expoptsetup,
                                       MSK_expoptimize,
                                       MSK_expoptfree)
    numcon = 1
    numvar = 3
    numter = 5
    subi = [0, 0, 0, 1, 1]
    subk = 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    c = np.array((40., 20., 40., 1/3., 4/3.))
    subj = [0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 1, 2]
    akj = np.array((-1, -.5, -1., 1., 1., 1., 1., 1., -2., -2., .5, -1.))
    numanz = 12
    objval = 0.
    xx = np.empty(3)
    y = np.empty(5)
    env = Env()
    expopttask = Task(env)
    expopthnd = None

    r = MSK_expoptsetup(expopttask,
                        0,
                        numcon,
                        numvar,
                        numter,
                        subi,
                        c,
                        subk,
                        subj,
                        akj,
                        numanz,
                        expopthnd)
    # verify that r == MSK_RES_OK
    
    r = MSK_expoptimize(expopttask,
                        prosta,
                        solsta,
                        objval,
                        xx,
                        y,
                        expopthnd)
    print 'objectove value is %s' % objval
    print 'primal variables are %s' % xx
    print 'dual variables are %s' % y


def ex2():
    # possibly more user-friendly version that could be enabled by a 
    # wrapper. Inspired by mosek MATLAB toolbox function mskgpopt
    # http://docs.mosek.com/7.0/toolbox/A_guided_tour.html
    from gpkit.mosek_interface import mskgpopt
    c = np.array([40, 20, 40, 1/3., 4/3.])
    # a will need to be sparse... todo: fix.
    a = np.array([[-1, -0.5, -1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [-2, -2, 0],
                  [0, 0.5, -1]])
    _map = [0, 0, 0, 1, 1]
    res = mskgpopt(c, a, _map)
    print res


if __name__ == '__main__':
    ex1()
    ex2()

