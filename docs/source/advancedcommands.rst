Advanced Commands
*****************

Substitutions
=============

Substitutions are a very general-purpose way to change every instance of one variable into either a number or another variable.

Substituting into a Posynomials, PosyArrays, and GPs
-----------------------------------------------------

The examples below all use Posynomials and PosyArrays, but the syntax is identical for GPs (except when it comes to sweep variables).

.. code-block:: python

    # from t_subs.py / t_NomialSubs / test_Basic
    x = Variable("x")
    p = x**2
    assert p.sub(x, 3) == 9
    assert p.sub(x.varkeys["x"], 3) == 9
    assert p.sub("x", 3) == 9

Here the variable `x` is being replaced with `3` in three ways: first by substituting for ``x`` directly, then by substituting for for the ``VarKey("x")``, then by substituting the string "x". In all cases the substitution is understood as being with the VarKey: when a variable is passed in the VarKey is pulled out of it, and when a string is passed in it is used as an argument to the posynomials ``varkeys`` dictionary.

Substituting multiple values
----------------------------

.. code-block:: python

    # from t_subs.py / t_NomialSubs / test_Vector
    x = Variable("x")
    y = Variable("y")
    z = VectorVariable(2, "z")
    p = x*y*z
    assert all(p.sub({x: 1, "y": 2}) == 2*z)
    assert all(p.sub({x: 1, y: 2, "z": [1, 2]}) == z.sub(z, [2, 4]))

To substitute in multiple variables, pass them in as a dictionary where the keys are what will be replaced and values are what it will be replaced with. Note that you can also substitute for VectorVariables by their name or by their PosyArray.

Substituting with nonnumeric values
-----------------------------------

You can also substitute in with sweep variables (for which see last week's FotW), strings, and monomials:

.. code-block:: python

    # from t_subs.py / t_NomialSubs

    def test_ScalarUnits(self):
        x = Variable("x", "m")
        xvk = x.varkeys.values()[0]
        descr_before = x.exp.keys()[0].descr
        y = Variable("y", "km")
        yvk = y.varkeys.values()[0]
        for x_ in ["x", xvk, x]:
            for y_ in ["y", yvk, y]:
                if not isinstance(y_, str) and type(xvk.descr["units"]) != str:
                    expected = 0.001
                else:
                    expected = 1.0
                self.assertAlmostEqual(expected, mag(x.sub(x_, y_).c))
        if type(xvk.descr["units"]) != str:
            z = Variable("z", "s")
            self.assertRaises(ValueError, y.sub, y, z)

Note that units are preserved, and that the value can be either a string (in which case it just renames the variable), a varkey (in which case it changes its description, including the name) or a Monomial (in which case it substitutes for the variable with a new monomial).

Substituting with replacement
------------------------------

Any of the substitutions above can be run with ``p.sub(*args, replace=True)`` to clobber any previously-substitued values.

Fixed Variables
---------------

When a GP is created, any fixed Variables are use to form a dictionary: ``{var: var.descr["value"] for var in self.varlocs if "value" in var.descr}``. This dictionary in then substituted into the GP's cost and constraints before the ``substitutions`` argument.

Substituting from a GP solution array
-------------------------------------

``gp.solution.subinto(p)`` will substitute the solution(s) for variables into the posynomial ``p``, returning a PosyArray. For a non-swept solution, this is equivalent to ``p.sub(gp.solution["variables"])``.

You can also substitute by just calling the solution, i.e. ``gp.solution(p)``. This returns a numpy array of just the coefficients (``c``) of the posynomial after substitution, and will raise a` `ValueError`` if some of the variables in ``p`` were not found in ``gp.solution``.


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
