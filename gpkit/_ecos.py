import ecos
import numpy as np
from scipy.sparse import csc_matrix
from gpkit.small_classes import CootMatrix
from gpkit import ConstraintSet
from gpkit.small_scripts import mag


def expcone_GP(gp):
    posynomials, varkeys = gp.posynomials, gp.varlocs
    vk_map = {vk: i for i, vk in enumerate(varkeys)}
    mon_idx, mon_map = 0, {}
    for posy in posynomials:
        for exp in posy.exps:
            if exp not in mon_map:
                mon_map[exp] = mon_idx
                mon_idx += 1

    n_vars = len(varkeys)
    n_posys = len(posynomials) - 1  # minus one for cost, the 0th posynomial
    n_monos = len(mon_map)

    dims = {"l": n_posys,  # number of positive orthant cone constraints
            "e": n_monos,  # number of exponential cone constraints
            "q": []}       # a number for each second order cone constraint

    # COST
    costvec = np.zeros(n_vars + n_monos)
    cost_posy = posynomials[0]
    cost_mon_idxs = [mon_map[exp] for exp in cost_posy.exps]
    for c, m_i in zip(cost_posy.cs, cost_mon_idxs):
    # C0*m
        costvec[n_vars+m_i] = mag(c)

    # POSITIVE ORTHANT CONE
    G, h = CootMatrix([], [], []), []
    for p_i, posy in enumerate(posynomials[1:]):
    # Ci*m <= 1
        mon_idxs = [mon_map[exp] for exp in posy.exps]
        for c, m_i in zip(posy.cs, mon_idxs):
            G.append(len(h), n_vars+m_i, c)  # Ci*m
        h.append(1.0)  # 1

    # EXPONENTIAL CONE
    #     per cone three rows of h-Gx become a, b, c
    #         exp(a/c) <= b/c, c > 0
    for exp, m_i in mon_map.items():  # NOTE: could be sorted
    # exp(Ei x) <= mu_i
        # a = Ei*x
        for vk, e in exp.items():
            G.append(len(h), vk_map[vk], -float(e))
        h.append(0.0)
        # b = mu_i
        G.append(len(h), n_vars+m_i, -1.0)
        h.append(0.0)
        # c = 1
        G.append(len(h), 0, 0.0)  # 0 row
        h.append(1.0)

    return costvec, G.tocsc(), np.array(h), dims


def ecoptimize_factory(gp):
    def ecoptimize(*args, **kwargs):
        c, G, h, dims = expcone_GP(gp)
        n_posys = len(gp.posynomials) - 1
        n_vars = len(gp.varlocs)
        opts = {'feastol': 1e-9, 'reltol': 1e-9, 'abstol': 1e-9, 'verbose': True}
        solution = ecos.solve(c, G, h, dims, **opts)
        status = solution["info"]["infostring"]
        print dict(zip(gp.varlocs, np.exp(solution["x"])))
        if status in ["Optimal solution found", "Close to optimal solution found"]:
            status = "optimal"
        return dict(status=status,
                    cost=solution["info"]["pcost"],
                    primal=solution["x"][:n_vars],
                    la=solution["z"][:n_posys]/solution["info"]["pcost"])

    return ecoptimize
