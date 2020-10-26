"""Plotting methods"""
from collections import Counter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from .plot_sweep import assign_axes
from ..repr_conventions import lineagestr
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


def treemap(model, sizebyconstraints=False):
    """Plots model structure as Plotly TreeMap

    Arguments
    ---------
    model: Model
        GPkit model object

    sizebyconstraints (optional): bool
        Whether to size blocks by number of constraints or use default sizing

    Returns
    -------
    plotly.graph_objects.Figure
        Plot of model hierarchy

    """
    modelnames = []
    parents = []
    numconstraints = []

    lineagestrs = []
    for constraint in model.flat():
        linstr = lineagestr(constraint.lineage)
        lineagestrs.append(linstr)

    modelcount = Counter(lineagestrs)
    for modelname, count in modelcount.items():
        modelnames.append(modelname)
        parent = modelname.rsplit(".", 1)[0]
        parents.append(parent)
        numconstraints.append(count)

    for parent in parents:
        if parent not in modelnames:
            modelnames.append(parent)
            if "." in parent:
                grandparent = parent.rsplit(".", 1)[0]
            else:
                grandparent = ""
            parents.append(grandparent)
            numconstraints.append(0)

    values = numconstraints if sizebyconstraints else None

    fig = go.Figure(go.Treemap(
        ids=modelnames,
        labels=[modelname.split(".")[-1] for modelname in modelnames],
        parents=parents,
        values=values,
    ))
    return fig

