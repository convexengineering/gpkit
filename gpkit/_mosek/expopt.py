# -*- coding: utf-8 -*-
"""Module for using the MOSEK EXPOPT C interface

    Example
    -------
    ``result = _mosek.expopt.imize(cs, A, p_idxs)``

    Raises
    ------
    ImportError
        If the local MOSEK library could not be loaded

"""
from __future__ import unicode_literals, print_function
from ctypes import pointer as ptr
from ctypes import c_double, c_int, CFUNCTYPE
from .baked_ctypesgen import load_library, String, c_void, POINTER, UNCHECKED
from .. import settings


class ModuleShortener(object):
    """Makes ctype calls look like C calls, but still use namespaces.

         example in C:  MSK_makeemptytask
       regular python:  MSK.MSK_makeemptytask
    w/ModuleShortener:  MSK._makeemptytask

    Arguments
    ---------
    stub : str
      String to append to all getattrs (the string "MSK" above)
    module : str
      Module to be shortened (the first "MSK" object above)
    """
    def __init__(self, stub, *modules):
        self.modules = modules
        self.stub = stub

    def __getattr__(self, attribute):
        """Appends stub to all getattr calls

        Args
        ----
        attribute : str
          Shortened stubless attribute name

        Returns
        -------
        attribute from self.module
        """
        for module in self.modules:
            try:
                return getattr(module, self.stub + attribute)
            except AttributeError:
                pass


# below is MSKsolsta_enum from mosek.h
#   positions changed as noted because MOSEK solves the dual GP problem
MSK_SOL_STA_LOOKUPTABLE = ["UNKNOWN",
                           "OPTIMAL",
                           "DUAL_FEAS",  # originally position 3
                           "PRIM_FEAS",  # originally position 2
                           "PRIM_AND_DUAL_FEAS",
                           "DUAL_INFEAS_CER",  # originally position 6
                           "PRIM_INFEAS_CER",  # originally position 5
                           "NEAR_OPTIMAL",
                           "NEAR_DUAL_FEAS",  # originally position 9
                           "NEAR_PRIM_FEAS",  # originally position 8
                           "NEAR_PRIM_AND_DUAL_FEAS",
                           "NEAR_DUAL_INFEAS_CER",  # originally position 12
                           "NEAR_PRIM_INFEAS_CER",  # originally position 11
                           "INTEGER_OPTIMAL",
                           "NEAR_INTEGER_OPTIMAL"]


def c_array(py_array, c_type):
    """Makes a C array from a python list or array and a C datatype

    Arguments
    ----------
      py_array: array-like data to convert
      c_type: C datatype to which elements of py_array will be converted

    Returns
    -------
    C array of chosen datatype
    """
    if not isinstance(py_array, list):
        pya = list(py_array)
    else:
        pya = py_array
    return (c_type * len(pya))(*pya)


MSK = ModuleShortener("MSK", load_library(settings["mosek_lib_path"]),
                      load_library(settings["mosek_gpkitbin_path"]))
MSK_RES_OK = 0
if settings["mosek_version"] == "7":
    MSK_IPAR_INTPNT_MAX_ITERATIONS = 28
    MSKuserhandle_t = POINTER(None)
else:
    MSK_IPAR_INTPNT_MAX_ITERATIONS = 19
    MSKuserhandle_t = POINTER(c_void)
MSKstreamfunc = CFUNCTYPE(UNCHECKED(None), MSKuserhandle_t, String)


@MSKstreamfunc
def printcb(_, msg):
    """Function to handle MOSEK's internal logging

    To enable printing to the python console, add a line like
        `print msg[:-1]`
    before the return statement.

    Arguments
    ----------
    void : None
      Placeholder to emulate C function
    msg : C string
      One particular log message; since it's a C string the last byte is null.

    Returns
    -------
    result : int
      0 indicates success
    """
    print(msg[:-1])
    return 0


# pylint: disable=unused-argument,too-many-locals,protected-access
def imize(c, A, p_idxs, *args, **kwargs):
    """Interface to the MOSEK EXPOPT solver via C

    This code is based on the example C file "tskexpopt.c" at
        "[...]/mosek/7/tools/examples/c/tstexpopt.c"

    Definitions
    -----------
    "[a,b] array of floats" indicates array-like data with shape [a,b]
    n is the number of monomials in the gp
    m is the number of variables in the gp
    p is the number of posynomials in the gp

    Arguments
    ----------
    c : floats array of shape n
        Coefficients of each monomial
    A: floats array of shape (m,n)
        Exponents of the various free variables for each monomial.
    p_idxs: ints array of shape n
        Posynomial index of each monomial

    Returns
    -------
    dict
        Contains the following keys
            "success": bool
            "objective_sol" float
                Optimal value of the objective
            "primal_sol": floats array of size m
                Optimal value of the free variables. Note: not in logspace.
            "dual_sol": floats array of size p
                Optimal value of the dual variables, in logspace.

    Raises
    ------
      None, but because it calls C code you can't Ctrl-C out of it easily. :-/

    """

    r = MSK_RES_OK

    numcon = 1+p_idxs[-1]
    numter, numvar = map(int, A.shape)

    xx = c_array([0]*numvar, c_double)
    yy = c_array([0]*numter, c_double)

    numcon, numvar, numter = map(c_int, [numcon, numvar, numter])

    c = c_array(c, c_double)
    subi = c_array(p_idxs, c_int)

    subk = c_array(A.row, c_int)
    subj = c_array(A.col, c_int)
    akj = c_array(A.data, c_double)
    numanz = c_int(len(A.data))

    objval = c_double()
    env = POINTER(c_void)()
    prosta = c_int()
    solsta = c_int()
    expopttask = POINTER(c_void)()
    expopthnd = POINTER(c_void)()
    # a little extra work to declare a pointer for expopthnd...
    ptr_expopthnd = POINTER(POINTER(c_void))(expopthnd)

    if r == MSK_RES_OK:
        r = MSK._makeenv(ptr(env), None)

    if r == MSK_RES_OK:
        r = MSK._makeemptytask(env, ptr(expopttask))

    if r == MSK_RES_OK:
        r = MSK._linkfunctotaskstream(expopttask, 0, None, printcb)

    if r == MSK_RES_OK:
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
                             # Pointer to data structure holding nonlinear data
                             ptr_expopthnd
                            )

    # Any parameter can now be changed with standard mosek function calls
    if r == MSK_RES_OK:
        r = MSK._putintparam(expopttask,
                             MSK_IPAR_INTPNT_MAX_ITERATIONS,
                             c_int(200))

    # Optimize,  xx holds the primal optimal solution,
    # yy holds solution to the dual problem
    if r == MSK_RES_OK:
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

    status = MSK_SOL_STA_LOOKUPTABLE[solsta.value]  # pylint:disable=invalid-sequence-index
    # Allow mosek's NEAR_DUAL_FEAS solution status, because our check in gp.py
    #   will catch solutions that don't actually meet our tolerance
    # TODO: when we standardize solver status responses, revisit this.
    if status == "NEAR_DUAL_FEAS":
        status = "OPTIMAL"
    return dict(status=status,
                objective=objval.value,
                primal=list(xx),
                nu=list(yy))
