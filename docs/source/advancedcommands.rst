Advanced Commands
*****************

Sweeps
======

Declaring Sweeps
----------------

Sweeps are useful for analyzing tradeoff surfaces. A sweep “value” is an Iterable of numbers, e.g. ``[1, 2, 3]``. Variables are swept when their substitution value takes the form ``('sweep', Iterable), (e.g. 'sweep', np.linspace(1e6, 1e7, 100))``. This can be done either during variable declaration (``x = Variable("x", ('sweep', [1, 2, 3])``) or during later substitution (``gp.sub("x", ('sweep', [1, 2, 3]))``, or if the variable was already substituted for a constant, ``gp.sub("x", ('sweep', [1, 2, 3]), replace=True))``.

Solving Sweeps
--------------

A GP with sweeps will solve for all possible combinations: e.g., if there’s a variable ``x`` with value ``('sweep', [1, 3])`` and a variable ``y`` with value ``('sweep', [14, 17])`` then the gp will be solved four times, for :math:`(x,y)\in\left\{(1, 14),\ (1, 17),\ (3, 14),\ (3, 17)\right\}`. The returned solutions will be a one-dimensional array (or 2-D for vector variables), accessed in the usual way.
Sweeping Vector Variables

Vector variables may also be substituted for: ``y = VectorVariable(3, "y", value=('sweep' ,[[1, 2], [1, 2], [1, 2]])`` will sweep :math:`y\ \forall~y_i\in\left\{1,2\right\}`.

Example Usage
-------------

.. code-block:: python

    # code from t_GPSubs.test_VectorSweep in tests/t_sub.py
    from gpkit import *

    x = Variable("x")
    y = VectorVariable(2, "y")
    gp = GP(x, [x >= y.prod()])
    gp.sub(y, ('sweep', [[2, 3], [5, 7, 11]]))
    a = gp.solve(printing=False)["cost"]
    b = [10, 14, 22, 15, 21, 33]
    assert all(abs(a-b)/(a+b) < 1e-7)
