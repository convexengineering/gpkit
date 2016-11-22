Building Complex Models
***********************


Inheriting from ``Model``
=========================

GPkit encourages an object-oriented modeling approach, where the modeler creates objects that inherit from ``Model`` to break large systems down into subsystems and analysis domains. The benefits of this approach include modularity, reusability, and the ability to more closely follow mental models of system hierarchy.

For example, two different models for a simple beam construction, designed by different modelers, could be used interchangeably inside another subsytem (such as an aircraft wing). 

At the end of this page a detailed example shows this technique in practice.


Vectorization
=============

``gpkit.Vectorize`` creates an environment in which Variables are created with an additional dimension:

.. code-block:: python

    "from gpkit/tests/t_vars.py"

    def test_shapes(self):
        with gpkit.Vectorize(3):
            with gpkit.Vectorize(5):
                y = gpkit.Variable("y")
                x = gpkit.VectorVariable(2, "x")
            z = gpkit.VectorVariable(7, "z")

        self.assertEqual(y.shape, (5, 3))
        self.assertEqual(x.shape, (2, 5, 3))
        self.assertEqual(z.shape, (7, 3))


This allows models written with scalar constraints to be created with vector constraints:

.. literalinclude:: examples/vectorize.py

.. literalinclude:: examples/vectorize_output.txt



Multipoint analysis modeling
============================

In many engineering models, there is a physical object that is operated in multiple conditions. Some variables correspond to the design of the object (size, weight, construction) while others are vectorized over the different conditions (speed, temperature, altitude). By combining named models and vectorization we can create intuitive representations of these systems while maintaining modularity and interoperability.

In the example below, the models ``Aircraft`` and ``Wing`` have a ``.dynamic()`` method which creates instances of ``AircraftPerformance`` and ``WingAero``, respectively. The ``Aircraft`` and ``Wing`` models create variables, such as size and weight without fuel, that represent a physical object. The ``dynamic`` models create properties which change based on the flight conditions, such as drag and fuel weight.

This means that when an aircraft is being optimized for a mission, you can create the aircraft (``AC`` in this example) and then pass it to a ``Mission`` model which can create vectorized aircraft performance models for each flight segment and flight condition.

.. literalinclude:: examples/performance_modeling.py

.. literalinclude:: examples/performance_modeling_output.txt
