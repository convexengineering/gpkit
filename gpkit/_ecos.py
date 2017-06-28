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

    # SECOND ORDER CONE
    #     for cone i,  dims["q"][i] rows of h-Gx  become t, s...
    #         ||s||_2 <= t
    if gp.robust is not None:
        # TODO: not entirely sure what this corresponds to uncertainty in...
        Sigma = gp.robust.get_Sigma()
        if gp.robust.distr == "lognormal":
            r_map = vk_map
        elif gp.robust.distr == "normal":
            # make auxiliary variables mu_r
            r_map = {r: n_monos+n_vars+i  # mon_map.get(r, i)
                     for i, r in enumerate(gp.robust.robustvarkeys)}
            costvec = np.concatenate((costvec, np.zeros(len(r_map))))
        for r in gp.robust.robustvarkeys:
        # better +1   Sigma*r_sigma*||\vec{log(r)}|| + log(r) <= log(r.median)
        # better -1   Sigma*r_sigma*||\vec{log(r)}|| + log(r.median) <= log(r)
            dims["q"].append(1 + len(gp.robust.robustvarkeys))
            r_sigma = gp.robust.substitutions[r.sigma]
            r_median = gp.robust.substitutions[r.median]
            if gp.robust.distr == "lognormal":
                r_median = np.log(r_median)
            # TODO: if r.median has been used as a variable, use that instead!
            # t = h_0-Gx_0 = (log(r.median)-log(r))*r.better
            # (better +1)  ||s|| + log(r) <= log(r.median)
            # (better -1)  ||s|| + log(r.median) <= log(r)
            G.append(len(h), r_map[r], r.better)
            h.append(r_median*r.better)
            for r_ in gp.robust.robustvarkeys:
                # s_n = h_n-Gx_n = Sigma*r.sigma * log(r_)
                G.append(len(h), r_map[r_], -Sigma*r_sigma)
                h.append(0)
        if gp.robust.distr == "normal":  # the exponential cone part
            for r in gp.robust.robustvarkeys:
            # exp(x_r) <= mu_r
                # a = log(r)
                G.append(len(h), vk_map[r], -1.0)
                h.append(0.0)
                # b = mu_r
                G.append(len(h), r_map[r], -1.0)
                h.append(0.0)
                # third row, c = 1
                G.append(len(h), 0, 0.0)  # empty G row
                h.append(1.0)
                dims["e"] += 1
    # NOTE: alternate way to get sensitivity to median is to create equality
    #       constraints for (x-x_median)/sigma, gives senss for sigma too...

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
        opts = {'max_iters': 1000,
                'feastol': 1e-9, 'reltol': 1e-9, 'abstol': 1e-9,
                'verbose': True}
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
