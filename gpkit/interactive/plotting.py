"""Plotting methods"""
from collections import Counter
import plotly.graph_objects as go
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


def treemap(model, itemize="variables", sizebycount=False):
    """Plots model structure as Plotly TreeMap

    Arguments
    ---------
    model: Model
        GPkit model object

    itemize (optional): string, either "variables" or "constraints"
        Specify whether to iterate over the model varkeys or constraints

    sizebycount (optional): bool
        Whether to size blocks by number of variables/constraints or use
        default sizing

    Returns
    -------
    plotly.graph_objects.Figure
        Plot of model hierarchy

    """
    modelnames = []
    parents = []
    sizes = []

    if itemize == "variables":
        lineagestrs = [l.lineagestr() or "Model" for l in model.varkeys]
    elif itemize == "constraints":
        lineagestrs = [l.lineagestr() or "Model" for l in model.flat()]

    modelcount = Counter(lineagestrs)
    for modelname, count in modelcount.items():
        modelnames.append(modelname)
        if "." in modelname:
            parent = modelname.rsplit(".", 1)[0]
        elif modelname != "Model":
            parent = "Model"
        else:
            parent = ""
        parents.append(parent)
        sizes.append(count)

    for parent in parents:
        if parent not in modelnames:
            modelnames.append(parent)
            if "." in parent:
                grandparent = parent.rsplit(".", 1)[0]
            elif parent != "Model":
                grandparent = "Model"
            else:
                grandparent = ""
            parents.append(grandparent)
            sizes.append(0)

    fig = go.Figure(go.Treemap(
        ids=modelnames,
        labels=[modelname.split(".")[-1] for modelname in modelnames],
        parents=parents,
        values=sizes if sizebycount else None,
    ))
    return fig
