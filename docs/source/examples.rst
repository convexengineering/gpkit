Examples
********

A Trivial GP
============
The most trivial GP we can think of:
minimize :math:`x` subject to the constraint :math:`x \ge 1`.

.. literalinclude:: examples/x_greaterthan_1.py

Of course, the optimal value is 1. Output:

.. literalinclude:: examples/x_greaterthan_1_output.txt

Maximizing the Volume of a Box
==============================
This example comes from Section 2.4 of the `GP tutorial <http://stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf>`_, by S. Boyd et. al.

.. literalinclude:: examples/simple_box.py

The output is

.. literalinclude:: examples/simple_box_output.txt

Water Tank
==========
.. literalinclude:: examples/water_tank.py

The output is 

.. literalinclude:: examples/water_tank_output.txt

.. Comments:

..
    .. literalinclude:: code/simple_box.py
        :language: python
        :emphasize-lines: 2-4, 6
        :lines: 1-7

iPython Notebook Examples
=========================

Also available on `nbviewer <http://nbviewer.ipython.org/github/convexopt/gpkit/tree/master/docs/source/ipynb/>`_.

    .. toctree::
       :maxdepth: 1

       ipynb/Box/Box.rst
       ipynb/Fuel/Fuel.rst


.. http://sphinx-doc.org/markup/code.html
