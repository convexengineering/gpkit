"""Plotting methods"""
import matplotlib.pyplot as plt
import numpy as np


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
