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


def _combine_nearby_ticks(ticks, lim, ntick=25):
    """This function deals with overlapping labels in non-regularly-spaced
    plotting ticks. Nearby ticks are combined (averaged), and their labels
    are concatenated into a single label.

    Arguments
    ---------
    ticks (list of (numeric, string) tuples):
        A list of matched tick values and labels
    lim (2-tuple) limits of this axis
    ntick (int): maximum number of ticks for which tick labels don't overlap

    Returns
    -------
    list, same format as 'ticks' input
    """
    min_spacing = (lim[1] - lim[0])/float(ntick - 1)
    ticks = sorted(ticks)   # sort by location; don't modify the original
    prevloc = float('-inf')
    decorated = []
    for t in ticks:
        # dist_to_left_nbr, tick, weight
        decorated.append([t[0] - prevloc, t, 1])
        prevloc = t[0]
    # hack together a doubly linked list -- add refs to left and right nghbrs
    for i in xrange(len(ticks)):
        decorated[i].extend([decorated[i-1] if i > 0 else None,
                             decorated[i+1] if i+1 < len(ticks) else None])
    while True: # will be terminated by break statement
        spacing_violations = [d for d in decorated if d[0] < min_spacing]
        if not spacing_violations:
            break
        # heuristic: deal with violations in order of severity
        spacing_violations.sort()
        for sv in spacing_violations:
            if sv[0] >= min_spacing:
                # previous tweaks solved an issue; recompute violations
                break
            _, (loc, name), weight, left, right = sv
            ldist, (lloc, lname), lweight, lleft, lright = left
            assert lright is sv
            # this tick is too close to left nghbr.
            # remove the left nghbr; recompute refs and distances
            # steps: 1. update this, 2: update lefts (2x); 3: update right
            newloc = (loc*weight + lloc*lweight)/float(weight + lweight)
            sv[0] = newloc - left[1][0] + ldist
            sv[1] = (newloc, ", ".join([lname, name]))
            sv[2] += lweight
            sv[3] = lleft
            left[:] = [float('inf'), None, 0, None, None]
            # be careful here, might be on a boundary
            if lleft is not None:
                lleft[4] = sv
            if right:
                right[0] = right[1][0] - newloc
    return [t for _, t, w, _, _ in decorated if w]


def sensitivity_plot(gp, keys=None, xmax=1, yxmax=1):
    """Plot percentage change in objective (cost) as variables change,
    using sensitivity information.

    Arguments
    ---------
    gp (GeometricProgram): a solved GP
    keys (iterable):
        list of variable keys to plot sensitivities for.
        None defaults to all gp substitutions.
    xmax (float):
        max (and min) percentage to plot on x axis
    yxmax (float):
        max (and min) percentage for y axis, *as a fraction of xmax*

    Returns
    -------
    matplotlib.pyplot Figure
    """
    if keys is None:
        keys = gp.substitutions.keys()
    right_ticks, top_ticks = [], []
    fig, left_ax = plt.subplots()
    ymax = yxmax*xmax
    left_ax.set_ylim((-ymax, ymax))
    for k in keys:
        s = gp.solution["sensitivities"]["variables"][k]
        left_ax.plot((-xmax, xmax), (-xmax*s, xmax*s))
        if abs(s) > yxmax:
            top_ticks.append((ymax/s, str(k)))
        else:
            right_ticks.append((xmax*s, str(k)))
    left_ax.set_xlabel("% change")
    left_ax.set_ylabel("Approx. % change in cost")
    left_ax.grid(True)

    # now make a right-hand y axis with text labels for each sensitivity key
    # might want to try Axes copy and ax.yaxis.tick_right()
    right_ax = left_ax.twinx()
    top_ax = left_ax.twiny()
    ylim = left_ax.get_ylim()
    right_ticks = _combine_nearby_ticks(right_ticks, lim=ylim)
    top_ticks = _combine_nearby_ticks(top_ticks, lim=(-xmax, xmax))
    right_ax.set_yticks([t[0] for t in right_ticks])
    right_ax.set_yticklabels([t[1] for t in right_ticks])
    top_ax.set_xticks([t[0] for t in top_ticks])
    top_ax.set_xticklabels([t[1] for t in top_ticks], rotation=90)
    right_ax.set_ylim(ylim)  # this needs to occur after set_yticks().
    top_ax.set_xlim(left_ax.get_xlim())
    return fig
