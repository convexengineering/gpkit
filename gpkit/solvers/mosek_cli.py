"""Module for using the MOSEK EXPOPT command line interface

    Example
    -------
    ``result = _mosek.cli_expopt.imize(cs, A, p_idxs, "gpkit_mosek")``

"""
import os
import shutil
import tempfile
import errno
import stat
from subprocess import check_output, CalledProcessError
from .. import settings
from ..exceptions import (UnknownInfeasible, InvalidLicense,
                          PrimalInfeasible, DualInfeasible)

def remove_read_only(func, path, exc):  # pragma: no cover
    "If we can't remove a file/directory, change permissions and try again."
    if func in (os.rmdir, os.remove) and exc[1].errno == errno.EACCES:
        # change the file to be readable,writable,executable: 0777
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        func(path)  # try again


def optimize_generator(path=None, **_):
    """Constructor for the MOSEK CLI solver function.

    Arguments
    ---------
    path : str (optional)
        The directory in which to put the MOSEK CLI input/output files.
        By default uses a system-appropriate temp directory.
    """
    tmpdir = path is None
    if tmpdir:
        path = tempfile.mkdtemp()
    filename = path + os.sep + "gpkit_mosek"
    if "mosek_bin_dir" in settings:
        if settings["mosek_bin_dir"] not in os.environ["PATH"]:
            os.environ["PATH"] += ":" + settings["mosek_bin_dir"]

    def optimize(*, c, A, p_idxs, **_):
        """Interface to the MOSEK "mskexpopt" command line solver

        Definitions
        -----------
        n is the number of monomials in the gp
        m is the number of variables in the gp
        p is the number of posynomials in the gp

        Arguments
        ---------
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
        write_output_file(filename, c, A, p_idxs)

        # run mskexpopt and print stdout
        solution_filename = filename + ".sol"
        try:
            for logline in check_output(["mskexpopt", filename, "-sol",
                                         solution_filename]).split(b"\n"):
                print(logline)
        except CalledProcessError as e:
            # invalid license return codes:
            #   expired: 233 (linux)
            #   missing: 240 (linux)
            if e.returncode in [233, 240]:  # pragma: no cover
                raise InvalidLicense() from e
            raise UnknownInfeasible() from e
        with open(solution_filename) as f:
            _, probsta = f.readline()[:-1].split("PROBLEM STATUS      : ")
            if probsta == "PRIMAL_INFEASIBLE":
                raise PrimalInfeasible()
            if probsta == "DUAL_INFEASIBLE":
                raise DualInfeasible()
            if probsta != "PRIMAL_AND_DUAL_FEASIBLE":
                raise UnknownInfeasible("PROBLEM STATUS: " + probsta)

            _, solsta = f.readline().split("SOLUTION STATUS     : ")
            # line looks like "OBJECTIVE           : 2.763550e+002"
            objective_val = float(f.readline().split()[2])
            assert_equal(f.readline(), "")
            assert_equal(f.readline(), "PRIMAL VARIABLES")
            assert_equal(f.readline(), "INDEX   ACTIVITY")
            primal_vals = read_vals(f)
            # read_vals reads the dividing blank line as well
            assert_equal(f.readline(), "DUAL VARIABLES")
            assert_equal(f.readline(), "INDEX   ACTIVITY")
            dual_vals = read_vals(f)

        if tmpdir:
            shutil.rmtree(path, ignore_errors=False, onerror=remove_read_only)

        return dict(status=solsta[:-1],
                    objective=objective_val,
                    primal=primal_vals,
                    nu=dual_vals)

    return optimize


def write_output_file(filename, c, A, p_idxs):
    "Writes a mosekexpopt compatible GP description to `filename`."
    with open(filename, "w") as f:
        numcon = p_idxs[-1]
        numter, numvar = map(int, A.shape)
        for n in [numcon, numvar, numter]:
            f.write("%d\n" % n)

        f.write("\n*c\n")
        f.writelines(["%.20e\n" % x for x in c])

        f.write("\n*p_idxs\n")
        f.writelines(["%d\n" % x for x in p_idxs])

        f.write("\n*t j A_tj\n")
        f.writelines(["%d %d %.20e\n" % tuple(x)
                      for x in zip(A.row, A.col, A.data)])


def assert_equal(received, expected):
    "Asserts that a file's next line is as expected."
    if expected.rstrip() != received.rstrip():  # pragma: no cover
        errstr = repr(expected)+" is not the same as "+repr(received)
        raise RuntimeWarning("could not read mskexpopt output file: "+errstr)


def read_vals(fil):
    "Read numeric values until a blank line occurs."
    vals = []
    line = fil.readline()
    while line not in ["", "\n"]:
        # lines look like "1       2.390776e+000   \n"
        vals.append(float(line.split()[1]))
        line = fil.readline()
    return vals
