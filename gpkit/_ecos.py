import ecos
import numpy as np
from scipy.sparse import csc_matrix
from gpkit.small_classes import CootMatrix
from gpkit import ConstraintSet
from gpkit.small_scripts import mag


def expcone_GP(posynomials, varkeys):
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

    # make costvec
    costvec = np.zeros(n_vars + n_monos)
    cost_posy = posynomials[0]
    cost_mon_idxs = [mon_map[exp] for exp in cost_posy.exps]
    for c, m_i in zip(cost_posy.cs, cost_mon_idxs):
        costvec[n_vars+m_i] = mag(c)  # c*m_i

    # make positive orthant constraints
    G, h = CootMatrix([], [], []), []
    for p_i, posy in enumerate(posynomials[1:]):
        mon_idxs = [mon_map[exp] for exp in posy.exps]
        for c, m_i in zip(posy.cs, mon_idxs):
            G.append(p_i, n_vars+m_i, c)  # c*m_i
        h.append(1.0)  # <= 1

    # make exp cone constraints
    for exp, m_i in mon_map.items():  # NOTE: could be sorted
        base = n_posys+3*m_i
        for vk, e in exp.items():
            G.append(base, vk_map[vk], -float(e))  # first row
        G.append(base+1, n_vars+m_i, -1.0)  # second row
        G.append(base+2, 0, 0.0)          # third row (fill out)
        h.extend([0.0, 0.0, 1.0])

    dims = {"l": n_posys, "e": n_monos}
    return costvec, G.tocsc(), np.array(h), dims


def ecoptimize_factory(gp):
    def ecoptimize(*args, **kwargs):
        c, G, h, dims = expcone_GP(gp.posynomials, gp.varlocs)
        n_posys = len(gp.posynomials) - 1
        n_vars = len(gp.varlocs)
        opts = {'feastol': 1e-9, 'reltol': 1e-9, 'abstol': 1e-9, 'verbose': True}
        solution = ecos.solve(c, G, h, dims, **opts)
        status = solution["info"]["infostring"]
        if status in ["Optimal solution found"]:
            status = "optimal"
        elif status != "Primal infeasible":
            raise ValueError(status+"\n"+str(dict(zip(gp.varlocs, np.exp(solution["x"])))))
        return dict(status=status,
                    cost=solution["info"]["pcost"],
                    primal=solution["x"][:n_vars],
                    la=solution["z"][:n_posys]/solution["info"]["pcost"])

    return ecoptimize
