Basic Commands
**************

Importing Modules
=================
The first thing to do when using GPkit is to import the classes and modules you will need. For example,

.. code-block:: python

    from gpkit import Variable, VectorVariable, GP


Declaring Variables
===================
Instances of the ``Variable`` class represent scalar decision variables. They store a key (i.e. name) used to look up the Variable in dictionaries, and optionally units, a description, and a value (if the Variable is to be held constant).


Decision Variables
------------------
.. code-block:: python

    # Declare a variable, x
    x = Variable('x')

    # Declare a variable, y, with units of meters
    y = Variable('y','m')

    # Declare a variable, z, with units of meters, and a description
    z = Variable('z', 'm', 'A variable called z with units of meters')

Note: make sure you have imported the class ``Variable`` beforehand.

Fixed Variables
---------------
To declare a variable with a constant value, use the ``Variable`` class, as above, but specify the ``value=`` input argument:

.. code-block:: python

    # Declare '\\rho' equal to 1.225 kg/m^3
    rho = Variable('\\rho', 1.225, 'kg/m^3', 'Density of air at sea level')

In the example above, the key name ``'\\rho'`` is for LaTeX printing (described later). The unit and description arguments are optional.

.. code-block:: python

    #Declare pi equal to 3.14
    pi = Variable('\\pi', 3.14)
    


Vector Variables
----------------
Vector variables are represented by the ``VectorVariable`` class.
The first argument is the length of the vector.
All other inputs follow those of the ``Variable`` class.

.. code-block:: python

    # Declare a 3-element vector variable 'x' with units of 'm'
    x = VectorVariable(3, "x", "m", "3-D Position")


Creating Monomials and Posynomials
==================================

Monomial and posynomial expressions can be created using mathematical operations on variables.
This is implemented under-the-hood using operator overloading in Python.

.. code-block:: python

    # create a Monomial term xy^2/z
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')
    m = x * y**2 / z
    type(m)  # gpkit.nomials.Monomial

.. code-block:: python

    # create a Posynomial expression x + xy^2
    x = Variable('x')
    y = Variable('y')
    p = x + x * y**2
    type(p)  # gpkit.nomials.Posynomial

Declaring Constraints
=====================

``Constraint`` objects represent constraints of the form ``Monomial >= Posynomial``  or ``Monomial == Monomial`` (which are the forms required for GP-compatibility).

Note that constraints must be formed using ``<=``, ``>=``, or ``==`` operators, not ``<`` or ``>``.

.. code-block:: python

    # consider a block with dimensions x, y, z less than 1
    # constrain surface area less than 1.0 m^2
    x = Variable('x', 'm')
    y = Variable('y', 'm')
    z = Variable('z', 'm')
    S = Variable('S', 1.0, 'm^2')
    c = (2*x*y + 2*x*z + 2*y*z <= S)
    type(c)  # gpkit.nomials.Constraint


Declaring Objective Functions
=============================
To declare an objective function, assign a Posynomial (or Monomial) to a variable name, such as ``objective``.

.. code-block:: python

    objective = 1/(x*y*z)

By convention, the objective is the function to be *minimized*. If you wish to *maximize* a function, take its reciprocal. For example, the code above creates an objective which, when minimized, will maximize ``x*y*z``.


Formulating a GP
================

The ``GP`` class represents a geometric programming problem. To create one, pass an objective and list of Constraints:

.. code-block:: python

    objective = 1/(x*y*z)
    constraints = [2*x*y + 2*x*z + 2*y*z <= S,
                   x >= 2*y]
    gp = GP(objective, constraints)


Solving the GP
==============

.. code-block:: python

    sol = gp.solve()


Printing Results
================

.. code-block:: python

    print sol.table()

.. code-block:: python

    print "The x dimension is %s." % (sol(x))
