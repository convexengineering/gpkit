"Implements plot_sweep1d function"
import matplotlib.pyplot as plt
from ..exceptions import InvalidGPConstraint


def assign_axes(var, posys, axes):
    "Assigns axes to posys, creating and formatting if necessary"
    if not hasattr(posys, "__iter__"):
        posys = [posys]
    N = len(posys)
    if axes is None:
        _, axes = plt.subplots(N, 1, sharex="col", figsize=(4.5, 3+1.5*N))
        if N == 1:
            axes = [axes]
        format_and_label_axes(var, posys, axes)
    elif N == 1 and not hasattr(axes, "__len__"):
        axes = [axes]
    return posys, axes


def format_and_label_axes(var, posys, axes, ylabel=True):
    "Formats and labels axes"
    for posy, ax in zip(posys, axes):
        if ylabel:
            if hasattr(posy, "key"):
                ylabel = (posy.key.descr.get("label", posy.key.name)
                          + " " + posy.key.unitstr(dimless="-"))
            else:
                ylabel = str(posy)
            ax.set_ylabel(ylabel)
        ax.grid(color="0.6")
        # ax.set_frame_on(False)
        for item in [ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(12)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(9)
        ax.tick_params(length=0)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in ax.spines.itervalues():
            i.set_linewidth(0.6)
            i.set_color("0.6")
            i.set_linestyle("dotted")
    xlabel = (var.key.descr.get("label", var.key.name)
              + " " + var.key.unitstr(dimless="-"))
    ax.set_xlabel(xlabel)  # pylint: disable=undefined-loop-variable
    plt.locator_params(nbins=4)
    plt.subplots_adjust(wspace=0.15)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def plot_1dsweepgrid(model, sweeps, posys, origsol=None, tol=0.01, **solveargs):
    """Creates and plots a sweep from an existing model

    Example usage:
    f, _ = plot_sweep_1d(m, {'x': np.linspace(1, 2, 5)}, 'y')
    f.savefig('mysweep.png')
    """
    origsubs = {swept: model.substitutions[swept] for swept in sweeps
                if swept in model.substitutions}
    if origsubs and not origsol:
        try:
            origsol = model.solve(**solveargs)
        except InvalidGPConstraint:
            origsol = model.localsolve(**solveargs)
    if not hasattr(posys, "__iter__"):
        posys = [posys]

    N, S = len(posys), len(sweeps)
    f, axes = plt.subplots(N, S, sharex='col', sharey='row',
                           figsize=(4+2*S, 4+2*N))
    plt.subplots_adjust(hspace=0.15)

    for i, (swept, swept_over) in enumerate(sweeps.items()):
        if isinstance(swept_over, tuple) and len(swept_over) == 2:
            sol = model.autosweep({swept: swept_over}, tol=tol, **solveargs)
        else:
            sol = model.sweep({swept: swept_over}, **solveargs)

        if len(sweeps) == 1:
            if len(posys) == 1:
                subaxes = [axes]
            else:
                subaxes = axes
        elif len(posys) == 1:
            subaxes = [axes[i]]
        else:
            subaxes = axes[:, i]

        sol.plot(posys, subaxes)
        if origsubs:
            for posy, ax in zip(posys, subaxes):
                ax.plot(origsubs[swept], origsol(posy), "ko", markersize=4)
        format_and_label_axes(swept, posys, subaxes, ylabel=(i is 0))
        model.substitutions.update(origsubs)

    return f, axes
