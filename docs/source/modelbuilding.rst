Building Complex Models
***********************

Checking for result changes
===========================
Tracking the effects of changes to complex models can get out of hand;
we recommend saving solutions with ``sol.save()``, then checking that new solutions are almost equivalent
with ``sol1.almost_equal(sol2)`` and/or ``print(sol1.diff(sol2))``, as shown below.

.. literalinclude:: examples/checking_result_changes.py

You can also check differences between swept solutions, or between
a point solution and a sweep.


Inheriting from ``Model``
=========================

GPkit encourages an object-oriented modeling approach, where the modeler creates objects that inherit from Model to break large systems down into subsystems and analysis domains. The benefits of this approach include modularity, reusability, and the ability to more closely follow mental models of system hierarchy. For example: two different models for a simple beam, designed by different modelers, should be able to be used interchangeably inside another subsystem (such as an aircraft wing) without either modeler having to write specifically with that use in mind.

When you create a class that inherits from Model, write a ``.setup()`` method to create the model's variables and return its constraints. ``GPkit.Model.__init__`` will call that method and automatically add your model's name and unique ID to any created variables.

Variables created in a ``setup`` method are added to the model even if they are not present in any constraints. This allows for simplistic 'template' models, which assume constant values for parameters and can grow incrementally in complexity as those variables are freed.

At the end of this page a detailed example shows this technique in practice.

Accessing Variables in Models
=============================
GPkit provides several ways to access a Variable in a ``Model`` (or ``ConstraintSet``):

- using ``Model.variables_byname(key)``. This returns all Variables in the Model, as well as in any submodels, that match the key.
- using ``Model.__getitem__``. ``Model[key]`` returns the only variable matching the key, even if the match occurs in a submodel. If multiple variables match the key, an error is raised.

These methods are illustrated in the following example.

.. literalinclude:: examples/model_var_access.py

.. literalinclude:: examples/model_var_access_output.txt
    :language: breakdowns

Vectorization
=============

``gpkit.Vectorize`` creates an environment in which Variables are created with an additional dimension:

.. literalinclude:: examples/vectorization.py

This allows models written with scalar constraints to be created with vector constraints:

.. literalinclude:: examples/vectorize.py

.. literalinclude:: examples/vectorize_output.txt
    :language: breakdowns



Multipoint analysis modeling
============================

In many engineering models, there is a physical object that is operated in multiple conditions. Some variables correspond to the design of the object (size, weight, construction) while others are vectorized over the different conditions (speed, temperature, altitude). By combining named models and vectorization we can create intuitive representations of these systems while maintaining modularity and interoperability.

In the example below, the models ``Aircraft`` and ``Wing`` have a ``.dynamic()`` method which creates instances of ``AircraftPerformance`` and ``WingAero``, respectively. The ``Aircraft`` and ``Wing`` models create variables, such as size and weight without fuel, that represent a physical object. The ``dynamic`` models create properties that change based on the flight conditions, such as drag and fuel weight.

This means that when an aircraft is being optimized for a mission, you can create the aircraft (``AC`` in this example) and then pass it to a ``Mission`` model which can create vectorized aircraft performance models for each flight segment and/or flight condition.

The :ref:`sensitivity diagram <sankey>` which this code outputs shows how it is organized (right-click and open in a new tab to see it more clearly):

.. figure:: figures/sankey_autosaves/Model.png

.. literalinclude:: examples/performance_modeling.py

Note that the output table has been filtered above to show only variables of interest.

.. literalinclude:: examples/performance_modeling_output.txt
    :language: breakdowns
