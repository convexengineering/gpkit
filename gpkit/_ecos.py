import ecos
import numpy as np
from scipy.sparse import csc_matrix
from gpkit.small_classes import CootMatrix
from gpkit import ConstraintSet
from gpkit.small_scripts import mag


def expcone_GP(gp):
    posynomials, varkeys = gp.posynomials, gp.varlocs
    n_vars = len(varkeys)
    n_posys = len(posynomials) - 1
    vk_map = {vk: i for i, vk in enumerate(varkeys)}
    mon_idx = 0
    mon_map = {}
    for posy in posynomials:
        for exp in posy.exps:
            if exp not in mon_map:
                mon_map[exp] = mon_idx
                mon_idx += 1
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

    # NOTE: alternate way to get sensitivity to median is to create equality
    #       constraints for (x-x_median)/sigma, gives senss for sigma too...
    # SECOND ORDER CONE
    #     for cone i,  dims["q"][i] rows of h - Gx are taken and become t, s...
    #     ||s||_2 <= t
    if gp.robust is not None:
        # TODO: not entirely sure what this corresponds to uncertainty in...
        R = len(gp.robust.robustvarkeys)
        Sigma = gp.robust.get_Sigma()
        for i, r in enumerate(gp.robust.robustvarkeys):
            dims["q"].append(1 + R)
            r_sigma = gp.robust.substitutions[r.sigma]
            r_median = gp.robust.substitutions[r.median]
            # TODO: if r.median has been used as a variable, use that instead!
            if gp.robust.distr == "lognormal":
            # +1   Sigma*r_sigma*||\vec{log(r)}|| + log(r) <= log(r.median)
            # -1   Sigma*r_sigma*||\vec{log(r)}|| + log(r.median) <= log(r)
                # t = h_0-Gx_0 = (log(r.median)-log(r))*r.better
                G.append(len(h), vk_map[r], r.better)
                h.append(np.log(r_median)*r.better)
                # (better +1)  ||s|| + log(r) <= log(r.median)
                # (better -1)  ||s|| + log(r.median) <= log(r)
                for r_ in gp.robust.robustvarkeys:
                    # s_n = h_n-Gx_n = Sigma*r.sigma * log(r_)
                    G.append(len(h), vk_map[r_], -Sigma*r_sigma)
                    h.append(0)
            elif gp.robust.distr == "normal":  # the second-order cone part
            # Sigma*r_sigma*||\vec{r}|| + r <= r.median
                # t = h_0-Gx_0 = (r.median - r)
                G.append(len(h), n_monos+n_vars+i, 1)
                h.append(r_median)
                # ||s|| + r <= r.median
                for j, r_ in enumerate(gp.robust.robustvarkeys):
                    # s_n = h_n-Gx_n = Sigma*r_sigma * r_
                    G.append(len(h), n_monos+n_vars+j, -Sigma*r_sigma)
                    h.append(0)
        if gp.robust.distr == "normal":  # the exponential cone part
        # exp(x_r) <= mu_r
            costvec = np.concatenate((costvec, np.zeros(R)))
            for i, r in enumerate(gp.robust.robustvarkeys):
                if gp.robust[r] in mon_map:
                    continue
                dims["e"] += 1
                # a = log(r)
                G.append(len(h), vk_map[r], -1.0)
                h.append(0.0)
                # b = mu_r
                G.append(len(h), n_monos+n_vars+i, -1.0)
                h.append(0.0)
                # third row, c = 1
                G.append(len(h), 0, 0.0)  # empty G row
                h.append(1.0)

    # EXPONENTIAL CONE
    #     per cone three rows of h-Gx are taken and become a, b, c
    #     exp(a/c) <= b/c, c > 0
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
        if status in ["Optimal solution found"]:
            status = "optimal"
        elif status != "Primal infeasible":
            raise ValueError(status+"\n"+str(dict(zip(gp.varlocs, np.exp(solution["x"])))))
        return dict(status=status,
                    cost=solution["info"]["pcost"],
                    primal=solution["x"][:n_vars],
                    la=solution["z"][:n_posys]/solution["info"]["pcost"])

    return ecoptimize
