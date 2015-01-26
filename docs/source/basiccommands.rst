Basic Commands
**************

Importing modules
=================
The first thing to do when using GPkit is to import the classes and modules you will need.

.. code-block:: python

    from gpkit import Variable, Vector Variable, GP


Declaring Variables
===================
The Variable class requires you to define a print string for the variable. It also gives you the option of defining units, a description, and a value (for constant parameters).

Decision Variables
------------------
.. code-block:: python

    # Declares a variable, x
    x = Variable('x')

    # Declares a variable, y with units of meters
    y = Variable('y','m')

    # Declares a variable, z with units of meters, and a description
    z = Variable('z', 'm', 'A variable called z with units of meters')

Hint: make sure you have imported the class ``Variable`` beforehand.

Parameters/Constants
--------------------
To declare a variable that has a constant value, simply create a variable using the Variable class, as above, but this time add the value of the parameter as an additional argument after the print string.

.. code-block:: python

    # Declares a constant
    # Print String: '\\rho'
    # Value: 1.225
    # Units: kg/m^3
    # Description: Density of air at sea level
    rho = Variable('\\rho', 1.225, 'kg/m^3', 'Density of air at sea level')

The unit and description arguments are optional.

.. code-block:: python

    #Declares pi
    pi = Variable('\\pi', 3.14)
    


Vector Variables
----------------
GPkit allows you to define vectors of variables. The first argument is the number

.. code-block:: python

    # Declares a vector variable
    # Number of elements: 3
    # Print string: "d"
    # Units: m
    # Description: "Dimension Vector"
    d   = VectorVariable(3, "d", "m", "Dimension Vector")


Declaring Constraints
=====================
Constraints are declared in a list format. This means they should be separated by comments and enclosed in square brackets ``[ ]``.

.. code-block:: python

    constraints = [ Re == (rho/mu)*V*(S/A)**0.5,
                    C_f == 0.074/Re**0.2,
                    W <= 0.5*rho*S*C_L*V**2,
                    W <= 0.5*rho*S*C_Lmax*V_min**2,
                    W >= W_0 + W_w,
                    W_w >= W_w_surf + W_w_strc
                  ]

You can add to your list of constraints using standard python list syntax:

.. code-block:: python

    constraints += [C_D >= C_D_fuse + C_D_wpar + C_D_ind]

Inequality constraints
----------------------

Standard python syntax is used for inequality symbols.


Equality constraints
--------------------

When declaring constraints it doesn't matter if they are in GP standard form or not. That is to say you could define the following constraint in either explicit or implicit form.

.. math::
    W = mg

.. code-block:: python

    W == m * g

.. math::
    \frac{W}{mg} = 1

.. code-block:: python

    W/(m * g) == 1


Declaring Objective Functions
=============================
Simple assign the objective function to a variable name, such as ``objective``.

.. code-block:: python

    objective = x

As is convention for optimization, the objective must be defined as the function that you want to *minimize*. So, if you want to *maximize* a function, you need to transform this into a minimization. With most optimization, this usually means throwing a minus sign in front of your objective function, but that isn't GP compatible. To transform things in a GP compatible way, take the reciprocal of the function you want to maximize. For example,

.. math::
    \text{maximize } x

is equivalent to

.. math::
    \text{minimize } \frac{1}{x}


Formulating the GP
==================
.. code-block:: python

    gp = GP(objective, constraints)


Solving the GP
==============

.. code-block:: python

    sol = gp.sol()


Printing Results
================

.. code-block:: python

    print sol.table()

.. code-block:: python

    print sol(x)
