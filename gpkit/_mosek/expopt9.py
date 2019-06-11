# -*- coding: utf-8 -*-
"""Module for using the MOSEK 9 Fusion interface

    Example
    -------
    ``result = _mosek.expopt9.imize(cs, A, p_idxs)``

    Raises
    ------
    ImportError
        If the local MOSEK library could not be loaded

"""

import os
import sys

try:
    from mosek import *  # pylint: disable=import-error
except Exception as e:
    raise ImportError("Could not load MOSEK library: "+repr(e))


import numpy as np

# Since the value of infinity is ignored, we define it solely
# for symbolic purposes
inf = 0.0

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def imize_factory(gp):
    def imize(**kwargs):
        numcostvars = 1  # first var is cost
        freevars = [v for v in gp.varkeys if v not in gp.substitutions]

        unique_exps = {}
        freevarlookup = {v: i+numcostvars for i, v in enumerate(freevars)}
        # vars:
        #   log(freevars)
        #   3 vars per unique monomial in posynomial
        #
        # constraints:
        #   linear constraints for monos and posys
        #   exponential cone constraints
        asubi = []
        asubj = []
        aval = []
        bkc = []
        blc = []
        buc = []

        # go through constraints
        #   find unique monomials
        #   establish monomial and posynomial linear constraints
        for constraintnum, hmap in enumerate(gp.hmaps):  # presuming cost is first posy
            if constraintnum == 0:  # -costvar, which will have diff logitude for posy/monomial costs...
                asubi.append(0)
                asubj.append(0)
                aval.append(-1.0)
            if len(hmap) == 1:
                (exp, c), = hmap.items()
                bkc.append(boundkey.up)
                blc.append(-inf)
                buc.append(-np.log(c))
                for var, x in exp.items():
                    asubi.append(constraintnum)
                    asubj.append(freevarlookup[var])
                    aval.append(x)
            else:
                bkc.append(boundkey.fx)
                blc.append(1.0)
                buc.append(1.0)
                for exp, c in hmap.items():
                    if (exp, c) not in unique_exps:
                        unique_exps[(exp, c)] = len(unique_exps)
                    asubi.append(constraintnum)
                    asubj.append(numcostvars + len(freevars) + 3*unique_exps[(exp, c)])
                    aval.append(1.0)
                    # if c >= 1e100:
                    #     raise ValueError(hmap)

        a2subi = []
        a2subj = []
        a2val = []
        b2kc = []
        b2lc = b2uc = []
        numposyslt1 = asubi[-1] + 1
        for (exp, c), exp_i in unique_exps.items():
            b2kc.append(boundkey.fx)
            b2uc.append(np.log(c))
            # z_i = log(exp)
            a2subi.append(numposyslt1 + exp_i)
            a2subj.append(numcostvars + len(freevars) + 3*exp_i + 2)
            a2val.append(1.0)
            for var, x in exp.items():
                a2subi.append(numposyslt1 + exp_i)
                a2subj.append(freevarlookup[var])
                a2val.append(-x)

        with Env() as env:
            with env.Task(0, 0) as task:
                task.set_Stream(streamtype.log, streamprinter)

                # Add variables and constraints
                task.appendvars(numcostvars + len(freevars) + 3*len(unique_exps))
                task.appendcons(numposyslt1 + 2*len(unique_exps))

                # Objective is the sum of three first variables
                task.putobjsense(objsense.minimize)
                task.putcslice(0, 1, [1.0])
                task.putvarboundsliceconst(0, numcostvars + len(freevars), boundkey.fr, -inf, inf)

                # Add the three linear constraints
                task.putaijlist(asubi, asubj, aval)
                task.putconboundslice(0, numposyslt1, bkc, blc, buc)

                # Add linear constraints for the expressions appearing in exp(...)
                task.putaijlist(a2subi, a2subj, a2val)
                task.putconboundslice(numposyslt1, numposyslt1+len(unique_exps), b2kc, b2lc, b2uc)

                expStart = numcostvars + len(freevars)
                numExp = len(unique_exps)
                # Add a single log-sum-exp constraint sum(log(exp(z_i))) <= 0
                # Assume numExp variable triples are ordered as (u0,t0,z0,u1,t1,z1...)
                # starting from variable with index expStart

                # sum(u_i) = 1 as constraint number c, u_i unbounded
                task.putvarboundlistconst(range(expStart, expStart + 3*numExp, 3),
                                          boundkey.fr, -inf, inf)

                # z_i unbounded
                task.putvarboundlistconst(range(expStart + 2, expStart + 2 + 3*numExp, 3),
                                          boundkey.fr, -inf, inf)

                # t_i = 1
                task.putvarboundlistconst(range(expStart + 1, expStart + 1 + 3*numExp, 3),
                                          boundkey.fx, 1.0, 1.0)

                # Every triple is in an exponential cone
                task.appendconesseq([conetype.pexp]*numExp, [0.0]*numExp, [3]*numExp, expStart)

                # Solve and map to original h, w, d
                task.optimize()
                primals = [0.0]*len(freevars)
                task.getxxslice(soltype.itr, numcostvars, numcostvars+len(freevars),
                                primals)

                lambdas = np.array([0.0]*numposyslt1)
                task.getyslice(soltype.itr, 0, numposyslt1, lambdas)

                solution_status = repr(task.getsolsta(soltype.itr))
                solution_status = solution_status.replace("solsta.", "")
                return dict(status=solution_status,
                            primal=primals,
                            la=-lambdas)
    return imize
