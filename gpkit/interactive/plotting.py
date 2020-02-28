"""Plotting methods"""
import matplotlib.pyplot as plt
import numpy as np
from .plot_sweep import assign_axes
from .. import GPCOLORS


def compare(models, sweeps, posys, tol=0.001):
    """Compares the values of posys over a sweep of several models.

    If posys is of the same length as models, this will plot different
    variables from different models.

    Currently only supports a single sweepvar.

    Example Usage:
    compare([aec, fbc], {"R": (160, 300)},
            ["cost", ("W_{\\rm batt}", "W_{\\rm fuel}")], tol=0.001)
    """
    sols = [m.autosweep(sweeps, tol, verbosity=0) for m in models]
    posys, axes = assign_axes(sols[0].bst.sweptvar, posys, None)
    for posy, ax in zip(posys, axes):
        for i, sol in enumerate(sols):
            if hasattr(posy, "__len__") and len(posy) == len(sols):
                p = posy[i]
            else:
                p = posy
            color = GPCOLORS[i % len(GPCOLORS)]
            if sol._is_cost(p):  # pylint: disable=protected-access
                ax.fill_between(sol.sampled_at,
                                sol.cost_lb(), sol.cost_ub(),
                                facecolor=color, edgecolor=color,
                                linewidth=0.75)
            else:
                ax.plot(sol.sampled_at, sol(p), color=color)


def plot_convergence(model):
    """Plots the convergence of a signomial programming model

    Arguments
    ---------
    model: Model
        Signomial programming model that has already been solved

    Returns
    -------
    matplotlib.pyplot Figure
        Plot of cost as functions of SP iteration #
    """
    fig, ax = plt.subplots()

    it = np.array([])
    cost = np.array([])
    for n in range(len(model.program.gps)):
        try:
            cost = np.append(cost, model.program.gps[n].result['cost'])
            it = np.append(it, n+1)
        except TypeError:
            pass
    ax.plot(it, cost, '-o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_xticks(range(1, len(model.program.gps)+1))
    return fig, ax
