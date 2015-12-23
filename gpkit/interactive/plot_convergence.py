"""Plotting the convergence of signomial programs"""
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
        Semilogy plot of variable values as functions of SP iteration #
    """
    newDict = {}

    fig, ax = plt.subplots()

    for key in model.program.gps[0].result['variables']:
        a = np.array([])
        for j in range(len(model.program.gps)):
            a = np.append(a, model.program.gps[j].result['variables'][key])
        newDict[key] = a
        ax.semilogy(np.arange(len(newDict[key])), newDict[key], label=key.name)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Normalized variable values')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig
