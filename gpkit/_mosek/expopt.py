from math import exp
from ctypes import CDLL
from ctypes import pointer as ptr
from ctypes import POINTER as ptr_factory
from ctypes import c_double, c_int, c_void_p
from os import sep as os_sep
from os.path import dirname as os_path_dirname


class module_shortener(object):
    """Makes ctype calls look like C calls, but still use namespaces.

          example in C:  MSK_makeemptytask
        regular python:  MSK.MSK_makeemptytask
    w/module_shortener:  MSK._makeemptytask
    """
    def __init__(self, stub, module):
        self.module = module
        self.stub = stub

    def __getattr__(self, attribute):
        return getattr(self.module, self.stub+attribute)


def c_array(py_array, c_type):
    "Makes a C array from a python list or array and a C datatype"
    if not isinstance(py_array, list):
        pya = list(py_array)
    else:
        pya = py_array
    return (c_type * len(pya))(*pya)


# Attempt to load MOSEK libraries
try:
    import lib.expopt_h as expopt_h
    MSK = module_shortener("MSK", expopt_h)
except Exception, e:
    raise ImportError("Could not load MOSEK library: "+repr(e))


# All streaming logs from MOSEK are passed to this function:
@MSK.streamfunc
def printcb(void, msg):
    #print msg[:-1]
    return 0


def imize(c, A, map_):
    "Solve a GP using MOSEK EXPOPT"

    r = MSK._RES_OK

    numcon = 1+map_[-1]
    numvar, numter = map(int, A.shape)

    xx = c_array([0]*numvar, c_double)
    yy = c_array([0]*numter, c_double)

    numcon, numvar, numter = map(c_int, [numcon, numvar, numter])

    c = c_array(c, c_double)
    subi = c_array(map_, c_int)

    subk = c_array(A.col, c_int)
    subj = c_array(A.row, c_int)
    akj = c_array(A.data, c_double)
    numanz = c_int(len(A.data))

    objval = c_double()
    env = MSK.env_t()
    prosta = MSK.prostae()
    solsta = MSK.solstae()
    expopttask = MSK.task_t()
    expopthnd = c_void_p()
    # a little extra work to declare a pointer for expopthnd...
    ptr_expopthnd = ptr_factory(c_void_p)(expopthnd)

    if r == MSK._RES_OK:
        r = MSK._makeenv(ptr(env), None)

    if r == MSK._RES_OK:
        r = MSK._makeemptytask(env, ptr(expopttask))

    if r == MSK._RES_OK:
        r = MSK._linkfunctotaskstream(expopttask,
                                      MSK._STREAM_LOG,
                                      None,
                                      printcb)

    if r == MSK._RES_OK:
        # Initialize expopttask with problem data
        r = MSK._expoptsetup(expopttask,
                             c_int(1),  # Solve the dual formulation
                             numcon,
                             numvar,
                             numter,
                             subi,
                             c,
                             subk,
                             subj,
                             akj,
                             numanz,
                             ptr_expopthnd
                             # Pointer to data structure holding nonlinear data
                             )

    # Any parameter can now be changed with standard mosek function calls
    if r == MSK._RES_OK:
        r = MSK._putintparam(expopttask,
                             MSK._IPAR_INTPNT_MAX_ITERATIONS,
                             c_int(200))

    # Optimize,  xx holds the primal optimal solution,
    # y holds solution to the dual problem if the dual formulation is used

    if r == MSK._RES_OK:
        r = MSK._expoptimize(expopttask,
                             ptr(prosta),
                             ptr(solsta),
                             ptr(objval),
                             ptr(xx),
                             ptr(yy),
                             ptr_expopthnd)

    # Free data allocated by expoptsetup
    if ptr_expopthnd:
        MSK._expoptfree(expopttask,
                        ptr_expopthnd)

    MSK._deletetask(ptr(expopttask))
    MSK._deleteenv(ptr(env))

    return dict(success=True,
                primal_sol=[exp(x) for x in xx],
                dual_sol=list(yy))
