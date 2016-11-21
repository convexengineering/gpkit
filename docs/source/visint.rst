Visualization and Interaction
*****************************

Plotting variable sensitivities
===============================

Sensitivities are a useful way to evaluate the tradeoffs in your model, as well as what aspects of the model are driving the solution and should be examined. To help with this, GPkit has an automatic sensitivity plotting function that can be accessed as follows:

.. code-block:: python

    from gpkit.interactive.plotting import sensitivity_plot
    sensitivity_plot(m)

Which produces the following plot:

.. figure::  sensitivities.png
   :width: 500 px

In this plot, steep lines that go up to the right are variables whose increase sharply increases (makes worse) the objective. Steep lines going down to the right are variables whose increase sharply decreases (improves) the objective.