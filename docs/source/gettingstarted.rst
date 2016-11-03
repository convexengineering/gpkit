Getting Started
***************

GPkit is a Python package, so we assume basic familiarity with Python: if you're new to Python we recommend you take a look at `Learn Python <http://www.learnpython.org>`_.

Otherwise, :ref:`install GPkit <installation>` and import away:

.. code-block:: python

    from gpkit import Variable, VectorVariable, Model


Declaring Variables
===================
Instances of the ``Variable`` class represent scalar variables. They create a ``VarKey`` to store the variable's name, units, a description, and value (if the Variable is to be held constant), as well as other metadata.


Free Variables
--------------
.. code-block:: python

    # Declare a variable, x
    x = Variable("x")

    # Declare a variable, y, with units of meters
    y = Variable("y", "m")

    # Declare a variable, z, with units of meters, and a description
    z = Variable("z", "m", "A variable called z with units of meters")

Fixed Variables
---------------
To declare a variable with a constant value, use the ``Variable`` class, as above, but put a number before the units:

.. code-block:: python

    # Declare \rho equal to 1.225 kg/m^3.
    # NOTE: in python string literals, backslashes must be doubled
    rho = Variable("\\rho", 1.225, "kg/m^3", "Density of air at sea level")

In the example above, the key name ``"\rho"`` is for LaTeX printing (described later). The unit and description arguments are optional.

.. code-block:: python

    #Declare pi equal to 3.14
    pi = Variable("\\pi", 3.14)



Vector Variables
----------------
Vector variables are represented by the ``VectorVariable`` class.
The first argument is the length of the vector.
All other inputs follow those of the ``Variable`` class.

.. code-block:: python

    # Declare a 3-element vector variable "x" with units of "m"
    x = VectorVariable(3, "x", "m", "Cube corner coordinates")
    x_min = VectorVariable(3, "x", [1, 2, 3], "m", "Cube corner minimum")


Creating Monomials and Posynomials
==================================

Monomial and posynomial expressions can be created using mathematical operations on variables.

.. code-block:: python

    # create a Monomial term xy^2/z
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")
    m = x * y**2 / z
    type(m)  # gpkit.nomials.Monomial

.. code-block:: python

    # create a Posynomial expression x + xy^2
    x = Variable("x")
    y = Variable("y")
    p = x + x * y**2
    type(p)  # gpkit.nomials.Posynomial

Declaring Constraints
=====================

.. Introduce ConstraintSets here

``Constraint`` objects represent constraints of the form ``Monomial >= Posynomial``  or ``Monomial == Monomial`` (which are the forms required for GP-compatibility).

Note that constraints must be formed using ``<=``, ``>=``, or ``==`` operators, not ``<`` or ``>``.

.. code-block:: python

    # consider a block with dimensions x, y, z less than 1
    # constrain surface area less than 1.0 m^2
    x = Variable("x", "m")
    y = Variable("y", "m")
    z = Variable("z", "m")
    S = Variable("S", 1.0, "m^2")
    c = (2*x*y + 2*x*z + 2*y*z <= S)
    type(c)  # gpkit.nomials.PosynomialInequality

Formulating a Model
================

The ``Model`` class represents an optimization problem. To create one, pass an objective and list of Constraints.

By convention, the objective is the function to be *minimized*. If you wish to *maximize* a function, take its reciprocal. For example, the code below creates an objective which, when minimized, will maximize ``x*y*z``.

.. code-block:: python

    objective = 1/(x*y*z)
    constraints = [2*x*y + 2*x*z + 2*y*z <= S,
                   x >= 2*y]
    m = Model(objective, constraints)


Solving the Model
=================

.. move example solve printouts (below) up to here

When solving the model you can change the level of information that gets printed to the screen with the ``verbosity`` setting. A verbosity of 1 (the default) prints warnings and timing; a verbosity of 2 prints solver output, and a verbosity of 0 prints nothing.

.. code-block:: python

    sol = m.solve(verbosity=0)


Printing Results
================

The solution object can represent itself as a table:

.. code-block:: python

    print sol.table()

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

.. code-block:: python

    print "The optimal value is %s." % sol["cost"]
    print "The x dimension is %s." % sol(x)
    print "The y dimension is %s." % sol["variables"]["y"]

::

    The optimal value is 15.5884619886.
    The x dimension is 0.5774 meter.
    The y dimension is 0.2887 meter.

.. refactor this section; explain what can be done with a SolutionArray
.. e.g. table(), __call__, ["variables"], etc.

Sensitivities and dual variables
================================

When a GP is solved, the solver returns not just the optimal value for the problem’s variables (known as the "primal solution") but also the effect that relaxing each constraint would have on the overall objective (the "dual solution").

From the dual solution GPkit computes the sensitivities for every fixed variable in the problem. This can be quite useful for seeing which constraints are most crucial, and prioritizing remodeling and assumption-checking.

Using variable sensitivities
----------------------------

Fixed variable sensitivities can be accessed from a SolutionArray’s ``["sensitivities"]["constants"]`` dict, as in this example:

.. code-block:: python

    import gpkit
    x = gpkit.Variable("x")
    x_min = gpkit.Variable("x_{min}", 2)
    sol = gpkit.Model(x, [x_min <= x]).solve()
    assert sol["sensitivities"]["constants"][x_min] == 1

These sensitivities are actually log derivatives (:math:`\frac{d \mathrm{log}(y)}{d \mathrm{log}(x)}`); whereas a regular derivative is a tangent line, these are tangent monomials, so the ``1`` above indicates that ``x_min`` has a linear relation with the objective. This is confirmed by a further example:

.. code-block:: python

    import gpkit
    x = gpkit.Variable("x")
    x_squared_min = gpkit.Variable("x^2_{min}", 2)
    sol = gpkit.Model(x, [x_squared_min <= x**2]).solve()
    assert sol["sensitivities"]["constants"][x_squared_min] == 2

.. add a plot of a monomial approximation vs a tangent approximation
