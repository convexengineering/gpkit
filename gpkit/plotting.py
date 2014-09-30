import matplotlib.pyplot as plt
import numpy as np

light_colors = ['#80cdc1']
dark_colors = ['#01665e']
neutral_dark = '#888888'
neutral_light = '#aaaaaa'


def contour_plot(ax, X, Y, Z, title, colors):
    light, dark = colors
    fmt = '%.4g'

    contf = ax.contour(X, Y, Z, 72, linewidths=0.5, colors=light)
    cont = ax.contour(X, Y, Z, 6, linewidths=1, colors=dark)

    plt.clabel(cont, fmt=fmt, colors=dark, fontsize=14)
    ax.set_title(title, color=dark, fontsize=14)
    ax.tick_params(colors=neutral_dark)
    ax.set_frame_on(False)
    ax.grid(which='both', color=neutral_light)


def contour_array(data, vardescrs, X, Y, Zs,
                  nrows, ncols, figsize,
                  xticks=None, yticks=None, colors=None):

    def get_label(obj):
        if isinstance(obj, dict):
            return obj.keys()[0], obj.values()[0]
        elif isinstance(obj, str):
            if not vardescrs[obj]:
                return "%s" % obj, data[obj]
            units, descr = vardescrs[obj]
            if units is None:
                return "%s %s" % (descr, obj), data[obj]
            else:
                return "%s %s [%s]" % (descr, obj, units), data[obj]
        else:
            raise TypeError(obj.__class__.__name__)

    xlabel, Xgrid = get_label(X)
    ylabel, Ygrid = get_label(Y)
    Z_lgs = [get_label(Z) for Z in Zs]

    if colors is None:
        colors = light_colors[0], dark_colors[0]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True)
    if nrows > 1 and ncols > 1:
        xlabeledaxes = axes[-1,:]
        ylabeledaxes = axes[:,0]
    elif ncols > 1:
        xlabeledaxes = axes
        ylabeledaxes = [axes[0]]
    else:
        xlabeledaxes = [axes[-1]]
        ylabeledaxes = axes

    for ax in xlabeledaxes:
        ax.set_xlabel(xlabel, color=neutral_dark)
        ax.set_xlim((Xgrid.min(), Xgrid.max()))
        if xticks is not None:
            ax.set_xticks(xticks, minor=True)
            m = len(xticks)
            major_xticks = [xticks[i] for i in (0, 1*m/4, 2*m/4, 3*m/4, m-1)]
            major_ticklabels = ["%.2g" % x for x in major_xticks]
            ax.set_xticks(major_xticks)
            ax.set_xticklabels(major_ticklabels)
    for ax in ylabeledaxes:
        ax.set_ylabel(ylabel, color=neutral_dark)
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
        if nrows > 1: row_vector = axes[i % nrows, :]
        else: row_vector = axes
        if ncols > 1: ax = row_vector[i % ncols]
        else: ax = row_vector[0]
        contour_plot(ax,
                     Xgrid, Ygrid, Zgrid, zlabel, colors)

    return fig, axes


def frontier_surface_plot(data, xvar, yvar, zvars,
                          colors=None, maxfigsize=(5,5), axsize=(5,5)):
    pass
