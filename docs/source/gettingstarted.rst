Getting Started with GPkit
**************************

GPkit is a Python package. We assume basic familiarity with Python. If you are new to Python take a look at `Learn Python <http://www.learnpython.org>`_.

GPkit is also a command line tool. This means that you need to be in the terminal (OS X/Linux) or command prompt (Windows) to use it. If you are not familiar with working in the command line, check out this `Learn Code the Hard Way tutorial <http://cli.learncodethehardway.org/book/>`_.

The first thing to do is `install GPkit <installation.html>`_ . Once you have done this, you can start using GPkit in 3 easy steps:

1. Open your command line interface (terminal/Command Prompt)
2. Open a Python interpreter. This can be done by typing ``python`` (or ``ipython`` if you have Anaconda and like colorful error messages).
3. Type ``import gpkit``

After doing this, your command line will look something like one of the following::

    $ python
    >>> import gpkit

    $ ipython
    In [1]: import gpkit

From here, you can use GPkit commands to formulate and solve geometric programs. To learn how to do this take a look at the `Basic Commands <basiccommands.html>`_.


Writing GPkit scripts
=====================
Another way to write and solve GPs is to write a scipt and save it as a .py file. To run this file (e.g. ``myscript.py``), type the following in your command line::

    $ python myscript.py

Again, ``ipython`` will also work here.


Basic Commands
==============

Importing modules
-----------------
The first thing to do when using GPkit is to import the classes and modules you will need.

.. code-block:: python

    from gpkit import Variable, Vector Variable, GP


Declaring Variables
-------------------


Decision Variables
^^^^^^^^^^^^^^^^^^
.. code-block:: python

	# Declares a variable called x with units of meters
    x = Variable('x', 'm', 'A variable called x with units of meters')

Hint: make sure you have imported the class ``Variable`` beforehand.


Parameters/Constants
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

	# Declares a constant
	# Constant Name: '\\phi'
	# Value: 42
	# Units: None
	# Description: The meaning of life
    phi = Variable('\\phi', 42, '_', 'The meaning of life')


Vector Variables
^^^^^^^^^^^^^^^^

.. code-block:: python

	# Declares a 3-element vector called d, with units of meters
    d   = VectorVariable(3, "d", "m", "Dimension Vector")

Declaring Constraints
---------------------
Constraints are declared in a list. This means they should be enclosed in square brackets ``[ ]``.



Inequality constraints
^^^^^^^^^^^^^^^^^^^^^^

Standard python syntax is used for inequality symbols.


Equality constraints
^^^^^^^^^^^^^^^^^^^^

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
-----------------------------
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
------------------
.. code-block:: python

    gp = GP(objective, constraints)


Solving the GP
--------------

.. code-block:: python

    sol = gp.sol()


Printing Results
----------------

.. code-block:: python

    print sol.table()

.. code-block:: python

    print sol(x)
