"Implements plot_sweep1d function"
import matplotlib.pyplot as plt
import numpy as np
from ..exceptions import InvalidGPConstraint
from ..tools import autosweep_1d

GPBLU = "#59ade4"


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def plot_sweep1d(model, sweeps, posys, axes=None, orig_sol=None, **solveargs):
    """Creates and plots a sweep from an existing model

    Example usage:
    f, _ = plot_sweep_1d(m, {'x': np.linspace(1, 2, 5)}, 'y')
    f.savefig('mysweep.png')
    """
    orig_subs = {swept: model.substitutions[swept] for swept in sweeps
                 if swept in model.substitutions}
    if orig_subs and not orig_sol:
        try:
            orig_sol = model.solve(**solveargs)
        except InvalidGPConstraint:
            orig_sol = model.localsolve(**solveargs)
    if not hasattr(posys, "__iter__"):
        posys = [posys]
    if axes:
        f = None
    else:
        N, S = len(posys), len(sweeps)
        f, axes = plt.subplots(N, S, sharex='col', sharey='row',
                               figsize=(2.5+2.5*S, 2.5+2.5*N))
        plt.subplots_adjust(hspace=0.1)

    for i, (swept, swept_over) in enumerate(sweeps.items()):
        if isinstance(swept_over, tuple) and len(swept_over) in [2, 3]:
            autosweep = True
            if len(swept_over) is 2:
                start, end = swept_over
                tol = 0.01
            else:
                start, end, tol = swept_over
            bst = autosweep_1d(model, tol, swept, [start, end], **solveargs)
            x = np.linspace(start, end, 100)
            sol = bst.sample_at(x)
        else:
            autosweep = False
            x = swept_over
            model.substitutions.update({swept: ('sweep', x)})
            try:
                sol = model.solve(**solveargs)
            except InvalidGPConstraint:
                sol = model.localsolve(**solveargs)
            del model.substitutions[swept]

        if len(sweeps) == 1:
            if len(posys) == 1:
                subaxes = [axes]
            else:
                subaxes = axes
        elif len(posys) == 1:
            subaxes = [axes[i]]
        else:
            subaxes = axes[:, i]
        for posy, ax in zip(posys, subaxes):
            if autosweep and (posy.exps == model.cost.exps and
                              posy.cs == model.cost.cs):
                # should look like a line plot
                ax.fill_between(x, sol.cost_lb(), sol.cost_ub(),
                                facecolor=GPBLU, edgecolor=GPBLU,
                                linewidth=0.75)
            else:
                ax.plot(x, sol(posy), color=GPBLU)
            if orig_subs:
                ax.plot(orig_subs[swept], orig_sol(posy), "ko",
                        markersize=2)
            if i == 0:
                if hasattr(posy, "key"):
                    ylabel = (posy.key.descr.get("label", posy.key.name)
                              + " " + posy.key.unitstr(dimless="-"))
                else:
                    ylabel = str(posy)
                ax.set_ylabel(ylabel)
            ax.grid(color="0.5")
            ax.set_frame_on(False)
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                item.set_fontsize(8)
            ax.tick_params(color="0.6")

        xlabel = (swept.key.descr.get("label", swept.key.name)
                  + " " + swept.key.unitstr(dimless="-"))
        ax.set_xlabel(xlabel)  # pylint: disable=undefined-loop-variable
        model.substitutions.update(orig_subs)

    return f, axes
