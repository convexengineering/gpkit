"""Plotting methods"""
import matplotlib.pyplot as plt
from ..nomials import VarKey
from ..small_scripts import unitstr


LIGHT_COLORS = ['#80cdc1']
DARK_COLORS = ['#01665e']
NEUTRAL_DARK = '#888888'
NEUTRAL_LIGHT = '#aaaaaa'


def contour_plot(ax, X, Y, Z, title, colors):
    light, dark = colors
    fmt = '%.4g'

    contf = ax.contour(X, Y, Z, 72, linewidths=0.5, colors=light)
    cont = ax.contour(X, Y, Z, 6, linewidths=1, colors=dark)

    plt.clabel(cont, fmt=fmt, colors=dark, fontsize=14)
    ax.set_title(title, color=dark, fontsize=14)
    ax.tick_params(colors=NEUTRAL_DARK)
    ax.set_frame_on(False)
    ax.grid(which='both', color=NEUTRAL_LIGHT)


def contour_array(data, X, Y, Zs,
                  nrows, ncols, figsize,
                  xticks=None, yticks=None, colors=None):

    def get_label(var):
        var = VarKey(var)
        label = var.name
        if "idx" in var.descr:
            idx = var.descr.pop("idx", None)
            var = VarKey(**var.descr)
            label += "_%s" % idx
            vals = data[var][:, idx]
        else:
            vals = data[var]
        if var.units:
            label += unitstr(var.units, " [%s] ")
        if "label" in var.descr:
            label += " %s" % var.descr["label"]
        return label, vals

    xlabel, Xgrid = get_label(X)
    ylabel, Ygrid = get_label(Y)
    Z_lgs = [get_label(Z) for Z in Zs]

    if colors is None:
        colors = LIGHT_COLORS[0], DARK_COLORS[0]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True)
    if nrows > 1 and ncols > 1:
        xlabeledaxes = axes[-1, :]
        ylabeledaxes = axes[:, 0]
    elif ncols > 1:
        xlabeledaxes = axes
        ylabeledaxes = [axes[0]]
    else:
        xlabeledaxes = [axes[-1]]
        ylabeledaxes = axes

    for ax in xlabeledaxes:
        ax.set_xlabel(xlabel, color=NEUTRAL_DARK)
        ax.set_xlim((Xgrid.min(), Xgrid.max()))
        if xticks is not None:
            ax.set_xticks(xticks, minor=True)
            m = len(xticks)
            major_xticks = [xticks[i] for i in (0, 1*m/4, 2*m/4, 3*m/4, m-1)]
            major_ticklabels = ["%.2g" % x for x in major_xticks]
            ax.set_xticks(major_xticks)
            ax.set_xticklabels(major_ticklabels)
    for ax in ylabeledaxes:
        ax.set_ylabel(ylabel, color=NEUTRAL_DARK)
        ax.set_ylim((Ygrid.min(), Ygrid.max()))
        if yticks is not None:
            ax.set_yticks(yticks, minor=True)
            m = len(yticks)
            major_yticks = [yticks[i] for i in (0, 1*m/4, 2*m/4, 3*m/4, m-1)]
            major_ticklabels = ["%.2g" % y for y in major_yticks]
            ax.set_yticks(major_yticks)
            ax.set_yticklabels(major_ticklabels)
    fig.tight_layout(h_pad=3)

    for i, Z_lg in enumerate(Z_lgs):
        zlabel, Zgrid = Z_lg
        row_vector = axes[i % nrows, :] if nrows > 1 else axes
        ax = row_vector[i % ncols] if ncols > 1 else row_vector[0]
        # hack begins
        Xgrid = Xgrid.reshape(len(xticks), len(yticks))
        Ygrid = Ygrid.reshape(len(xticks), len(yticks))
        Zgrid = Zgrid.reshape(len(xticks), len(yticks))
        # hack ends
        contour_plot(ax, Xgrid, Ygrid, Zgrid, zlabel, colors)

    return fig, axes


def frontier_surface_plot(data, xvar, yvar, zvars,
                          colors=None, maxfigsize=(5,5), axsize=(5,5)):
    pass


def plot_frontiers(gp, Zs, x=1, y=3, figsize=(15,5)):
    "Helper function to plot 2d contour plots."
    sol = gp.solution
    data = dict(sol["variables"])
    data.update({"S{%s}" % k: v
                 for (k, v) in sol["sensitivities"]["variables"].items()})
    if len(gp.sweep) == 2:
        contour_array(data,
                      gp.sweep.keys()[0],
                      gp.sweep.keys()[1],
                      Zs, x, y, figsize,
                      xticks=gp.sweep.values()[0],
                      yticks=gp.sweep.values()[1])


def sensitivity_plot(gp, keys=None, xlim=(-1, 1)):
    """Plot percentage change in objective (cost) as variables change,
    using sensitivity information.

    Arguments
    ---------
    gp (GeometricProgram): a solved GP
    keys (iterable):
        list of variable keys to plot sensitivities for.
        None defaults to all gp substitutions.
    xlim (2-tuple): min and max x percentages to plot

    Returns
    -------
    matplotlib.pyplot Figure
    """
    if keys is None:
        keys = gp.substitutions.keys()
    sens_dict = gp.solution["sensitivities"]["variables"]
    # set up for sorting (will need this later)
    named_sensitivities = [(sens_dict[k], str(k)) for k in keys]
    names = []
    ticks = []
    fig, left_ax = plt.subplots()
    for s, name in named_sensitivities:
        right_side_val = xlim[1]*s
        left_ax.plot(xlim, (xlim[0]*s, right_side_val))
        names.append(name)
        ticks.append(right_side_val)

    # now make a right-hand y axis with text labels for each sensitivity key
    right_ax = left_ax.twinx()
    right_ax.set_ylim(left_ax.get_ylim())
    right_ax.set_yticks(ticks)
    right_ax.set_yticklabels(names)
    left_ax.set_xlabel("Percentage change")
    left_ax.set_ylabel("Percentage change in cost")
    return fig
