# -*- coding: utf-8 -*-
"""Module for using the MOSEK EXPOPT command line interface

    Example
    -------
    ``result = _mosek.cli_expopt.imize(cs, A, p_idxs, "gpkit_mosek")``

"""

import os
import shutil
import errno
import stat
from math import exp
from subprocess import check_output
from .. import settings


def errorRemoveReadonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        # change the file to be readable,writable,executable: 0777
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        # retry
        func(path)
    else:
        pass


def imize_fn(filename):
    folder = os.path.expanduser("~") + os.sep + "gpkit_tmp"
    filename = folder + os.sep + filename
    os.environ['PATH'] = (os.environ['PATH'] + ':%s' %
                          settings["mosek_bin_dir"][0])

    def imize(c, A, p_idxs, k):
        """Interface to the MOSEK "mskexpopt" command line solver

        Definitions
        -----------
        "[a,b] array of floats" indicates array-like data with shape [a,b]
        n is the number of monomials in the gp
        m is the number of variables in the gp
        p is the number of posynomials in the gp

        Parameters
        ----------
        c : floats array of shape n
            Coefficients of each monomial
        A: floats array of shape (m,n)
            Exponents of the various free variables for each monomial.
        p_idxs: ints array of shape n
            Posynomial index of each monomial
        filename: str
            Filename prefix for temporary files

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
        RuntimeWarning
            If the format of mskexpopt's output file is unexpected.

        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(filename, "w") as f:
            numcon = 1+p_idxs[-1]
            numter, numvar = map(int, A.shape)
            for n in [numcon, numvar, numter]:
                f.write("%d\n" % n)

            f.write("\n*c\n")
            f.writelines(["%.20e\n" % x for x in c])

            f.write("\n*p_idxs\n")
            f.writelines(["%d\n" % x for x in p_idxs])

            t_j_Atj = zip(A.row, A.col, A.data)

            f.write("\n*t j A_tj\n")
            f.writelines(["%d %d %.20e\n" % tuple(x)
                          for x in t_j_Atj])

        check_output(["mskexpopt", filename])

        with open(filename+".sol") as f:
            assert_line(f, "PROBLEM STATUS      : PRIMAL_AND_DUAL_FEASIBLE\n")
            assert_line(f, "SOLUTION STATUS     : OPTIMAL\n")
            # line looks like "OBJECTIVE           : 2.763550e+002"
            objective_val = float(f.readline().split()[2])
            assert_line(f, "\n")
            assert_line(f, "PRIMAL VARIABLES\n")
            assert_line(f, "INDEX   ACTIVITY\n")
            primal_vals = list(read_vals(f))

            assert_line(f, "DUAL VARIABLES\n")
            assert_line(f, "INDEX   ACTIVITY\n")
            dual_vals = read_vals(f)

        shutil.rmtree(folder, ignore_errors=False,
                      onerror=errorRemoveReadonly)

        return dict(status="optimal",
                    objective=objective_val,
                    primal=primal_vals,
                    nu=dual_vals)

    return imize


def assert_line(f, expected):
    received = f.readline()
    if tuple(expected[:-1].split()) != tuple(received[:-1].split()):
        errstr = repr(expected)+" is not the same as "+repr(received)
        raise RuntimeWarning(errstr)


def read_vals(f):
    vals = []
    line = f.readline()
    while line not in ["", "\n"]:
        # lines look like "1       2.390776e+000   \n"
        vals.append(float(line.split()[1]))
        line = f.readline()
    return vals
