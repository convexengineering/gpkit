import os
from math import exp
from subprocess import check_output


def imize(c, A, map_, filename):
    if not os.path.exists("gpkit_tmp"):
        os.makedirs("gpkit_tmp")

    filename = "gpkit_tmp" + os.sep + filename
    with open(filename, 'w') as f:
        numcon = 1+map_[-1]
        numter, numvar = map(int, A.shape)
        for n in [numcon, numter, numvar]:
            f.write("%d\n" % n)

        f.write("\n*c\n")
        f.writelines(["%.20e\n" % x for x in c])

        f.write("\n*map_\n")
        f.writelines(["%d\n" % x for x in map_])

        t_j_Atj = zip(A.col, A.row, A.data)

        f.write("\n*t j A_tj\n")
        f.writelines(["%d %d %.20e\n" % tuple(x)
                      for x in t_j_Atj])

    check_output("mskexpopt "+filename, shell=True)

    with open(filename+".sol") as f:
        assert_line(f, "PROBLEM STATUS      : PRIMAL_AND_DUAL_FEASIBLE\n")
        assert_line(f, "SOLUTION STATUS     : OPTIMAL\n")
        # line looks like "OBJECTIVE           : 2.763550e+002"
        objective_val = float(f.readline().split()[2])
        assert_line(f, "\n")
        assert_line(f, "PRIMAL VARIABLES\n")
        assert_line(f, "INDEX   ACTIVITY\n")
        primal_vals = map(exp, read_vals(f))

        assert_line(f, "DUAL VARIABLES\n")
        assert_line(f, "INDEX   ACTIVITY\n")
        dual_vals = read_vals(f)

    os.remove(filename)
    os.remove(filename+".sol")
    os.removedirs("gpkit_tmp")
    return dict(success=True,
                objective_sol=objective_val,
                primal_sol=primal_vals,
                dual_sol=dual_vals)


def assert_line(f, expected):
    received = f.readline()
    if tuple(expected[:-1].split()) != tuple(received[:-1].split()):
        errstr = repr(expected)+" is not the same as "+repr(received)
        raise Exception(errstr)


def read_vals(f):
    vals = []
    while True:
        line = f.readline()
        if line == "\n" or line == "":
            break
        else:
            # lines look like "1       2.390776e+000   \n"
            vals.append(float(line.split()[1]))
    return vals
