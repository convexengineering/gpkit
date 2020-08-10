Getting Started
***************

GPkit is a Python package, so we assume basic familiarity with Python: if you're new to Python we recommend you take a look at `Learn Python <http://www.learnpython.org>`_.

Otherwise, :ref:`install GPkit <installation>` and import away:

.. literalinclude:: examples/gettingstarted.py
    :lines: 2-3

Declaring Variables
===================
Instances of the ``Variable`` class represent scalar variables. They create a ``VarKey`` to store the variable's name, units, a description, and value (if the Variable is to be held constant), as well as other metadata.


Free Variables
--------------
.. literalinclude:: examples/gettingstarted.py
    :lines: 6-13

Fixed Variables
---------------
To declare a variable with a constant value, use the ``Variable`` class, as above, but put a number before the units:

.. literalinclude:: examples/gettingstarted.py
    :lines: 16

In the example above, the key name ``"\rho"`` is for LaTeX printing (described later). The unit and description arguments are optional.

.. literalinclude:: examples/gettingstarted.py
    :lines: 19-20

Vector Variables
----------------
Vector variables are represented by the ``VectorVariable`` class.
The first argument is the length of the vector.
All other inputs follow those of the ``Variable`` class.

.. literalinclude:: examples/gettingstarted.py
    :lines: 23-25

Creating Monomials and Posynomials
==================================

Monomial and posynomial expressions can be created using mathematical operations on variables.

.. literalinclude:: examples/gettingstarted.py
    :lines: 28-33

.. literalinclude:: examples/gettingstarted.py
    :lines: 36-40

Declaring Constraints
=====================

.. Introduce ConstraintSets here

``Constraint`` objects represent constraints of the form ``Monomial >= Posynomial``  or ``Monomial == Monomial`` (which are the forms required for GP-compatibility).

Note that constraints must be formed using ``<=``, ``>=``, or ``==`` operators, not ``<`` or ``>``.

.. literalinclude:: examples/gettingstarted.py
    :lines: 43-50

Formulating a Model
================

The ``Model`` class represents an optimization problem. To create one, pass an objective and list of Constraints.

By convention, the objective is the function to be *minimized*. If you wish to *maximize* a function, take its reciprocal. For example, the code below creates an objective which, when minimized, will maximize ``x*y*z``.

.. literalinclude:: examples/gettingstarted.py
    :lines: 53-60

Solving the Model
=================

.. move example solve printouts (below) up to here

When solving the model you can change the level of information that gets printed to the screen with the ``verbosity`` setting. A verbosity of 1 (the default) prints warnings and timing; a verbosity of 2 prints solver output, and a verbosity of 0 prints nothing.

.. literalinclude:: examples/gettingstarted.py
    :lines: 63

Printing Results
================

The solution object can represent itself as a table:

.. literalinclude:: examples/gettingstarted.py
    :lines: 66

::

    Cost
    ----
    15.59 [1/m**3]

    Free Variables
    --------------
    x : 0.5774  [m]
    y : 0.2887  [m]
    z : 0.3849  [m]

    Constants
    ---------
    S : 1  [m**2]

    Sensitivities
    -------------
    S : -1.5

We can also print the optimal value and solved variables individually.

.. literalinclude:: examples/gettingstarted.py
    :lines: 69

::

    The optimal value is 15.5884619886.
    The x dimension is 0.5774 meter.
    The y dimension is 0.2887 meter.

.. refactor this section; explain what can be done with a SolutionArray
.. e.g. table(), __call__, ["variables"], etc.

Sensitivities and Dual Variables
================================

When a GP is solved, the solver returns not just the optimal value for the problem’s variables (known as the "primal solution") but also the effect that relaxing each constraint would have on the overall objective (the "dual solution").

From the dual solution GPkit computes the sensitivities for every fixed variable in the problem. This can be quite useful for seeing which constraints are most crucial, and prioritizing remodeling and assumption-checking.

Using Variable Sensitivities
----------------------------

Fixed variable sensitivities can be accessed from a SolutionArray’s ``["sensitivities"]["variables"]`` dict, as in this example:

.. literalinclude:: examples/gettingstarted.py
    :lines: 72-75

These sensitivities are actually log derivatives (:math:`\frac{d \mathrm{log}(y)}{d \mathrm{log}(x)}`); whereas a regular derivative is a tangent line, these are tangent monomials, so the ``1`` above indicates that ``x_min`` has a linear relation with the objective. This is confirmed by a further example:

.. literalinclude:: examples/gettingstarted.py
    :lines: 77-80

.. add a plot of a monomial approximation vs a tangent approximation
