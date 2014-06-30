import os
import shutil
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
        assert f.readline() == "PROBLEM STATUS      : PRIMAL_AND_DUAL_FEASIBLE\n"
        assert f.readline() == "SOLUTION STATUS     : OPTIMAL\n"
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        vals = []
        for line in f:
            if line == "\n": break
            else:
                idx, val = line.split()
                vals.append(exp(float(val)))
        return vals

    shutil.removetree("gpkit_tmp")
