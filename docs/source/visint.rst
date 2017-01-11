Visualization and Interaction
*****************************

Plotting a 1D Sweep
==================

A function called ``plot_sweep1d`` has been created to facilitate creating, solving, and plotting the results of a single-variable sweep. Example usage is as follows:

.. literalinclude:: examples/plot_sweep1d.py

The result looks like:

.. code-block:: text

    Solving over 20 passes.
    Sweeping took 0.34 seconds.

.. figure:: examples/plot_sweep1d.png
    :align: center

.. code-block:: text

    Solved after 7 passes.
    Possible log error +/-0.000476
    Autosweeping took 0.11 seconds.

.. figure:: examples/plot_autosweep1d.png
    :align: center


Plotting variable sensitivities
===============================

Sensitivities are a useful way to evaluate the tradeoffs in your model, as well as what aspects of the model are driving the solution and should be examined. To help with this, ``gpkit.interactive`` has an automatic sensitivity plotting function that can be accessed as follows:

.. code-block:: python

    from gpkit.interactive.plotting import sensitivity_plot
    sensitivity_plot(m)

Which produces the following plot:

.. figure::  sensitivities.png
   :align: center

In this plot, steep lines that go up to the right are variables whose increase sharply increases (makes worse) the objective. Steep lines going down to the right are variables whose increase sharply decreases (improves) the objective. Only local sensitivities are displayed, so the lines are optimistic: the real effect of changing parameters may lead to higher costs than shown.
