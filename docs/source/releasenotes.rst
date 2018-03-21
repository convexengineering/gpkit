Release Notes
*************

This page lists the changes made in each point version of gpkit.

Version 0.7.0
=============
 * Variable's values are now used only in their first ConstraintSet; in other ConstraintSets they're free variables
 * Variable values can be preserved by setting ``constant=True`` during variable declaration
 * MOSEK home directory can be set by a ``MSKHOME`` environment variable at build time
 * ``sol(var)`` now always returns Pint Quantities, even if the variable is dimensionless
 * ``sol[...][var]``, on the other hand, now always returns floats / numpy arrays of floats
 * Optional boundedness checking in docstring (see usage in `docs <http://gpkit.readthedocs.io/en/latest/modelbuilding.html#multipoint-analysis-modeling>`_)
 * Automatic boundedness checking for GPs
 * `Sankey diagrams <http://gpkit.readthedocs.io/en/latest/visint.html>`_
 * Many other fixes

Version 0.6.0
=============
 * new much faster NomialMap data structure (#682)
 * Many bug fixes and small improvements.
 * 6900 lines of code, 2200 lines of tests, 2100 lines of docstring.

Version 0.5.3
=============
 * faster SP solves (#1109)
 * LinkedConstraintSet deprecated (#1110)
 * Fixes to autosweep, ConstraintSet, interactive
 * Solution time is now stored with soltutions (including sweeps/SPs)
 * Model strings are divided with slashes (e.g. Airplane/Wing)

Version 0.5.2
=============
 * Added new ``sweep`` and ``autosweep`` methods to Model
    * Added ``plot`` routines to the results of those routines to make it easy to plot a 1D sweep.
 * Added new ``summary`` method to solution_array.
    * It and table accept iterables of vars, will only print vars in that iterable (or, by default, all vars)
 * Cleaned up and documented the ``interactive`` submodule
    * removed contour and sensitivity plots
    * added a 1D-sweep plotting function
    * added that plotting function as an option within the control panel interface
 * Overhauled and documented three types of variables whose value is determined by functions:
    * calculated constants
    * post-solve calculated variables
    * between-GP-solves calculated variables (for Sequential Geometric Programs)
 * Fix ``Bounded`` and implement ``debug()`` for SPs
 * Apply ``subinplace`` to substitutions dictionary as well
 * Require GP substitutions to be Numbers only
 * Extend Bounded to one-sided bounds
 * Print model's numbers by default, unless ``"modelnums" in exclude``
 * Implement lazy keymapping, allowing GP/SP results to be KeyDicts
 * Handle Signomial Inequalities that become Posynomial Inequalities after substitution
 * Various documentation updates
 * Various bug fixes

Version 0.5.1
=============
 * O(N) sums and monomial products
 * Warn about invalid ConstraintSet elements
 * allow setting Tight tolerance as a class attribute
 * full backwards compatibility for __init__ methods
 * scripts to test remote repositories
 * minor fixes, tests, and refactors
 * 3550 lines of code, 1800 lines of tests, 1700 lines of docstring. (not counting `interactive`)

Version 0.5.0
=============
 * No longer recommend the use of linked variables and subinplace (see below)
 * Switched default solver to MOSEK
 * Added Linked Variable diagram (PR #915)
 * Changed how overloaded operators interact with pint (PR #938)
 * Added and documented debugging tools (PR #933)
 * Added and documented vectorization tools
 * Documented modular model construction
 * 3200 lines of code, 1800 lines of tests, 1700 lines of docstring. (not counting `interactive`)

Changes to named models / Model inheritance
-------------------------------------------
We are deprecating the creation of named submodels with custom ``__init__`` methods. Previously, variables created during ``__init__`` in any class inheriting from Model were replaced by a copy with  ``__class__.__name__`` added as varkey metadata. This was slow, a bit irregular, and hacky.

We're moving to an explicitly-irregular ``setup`` method, which (if declared for a class inheriting from Model) is automatically called during ``Model.__init__`` inside a ``NamedVariables(self.__class__.__name__)`` environment. This 1) handles the naming of variables more explicitly and efficiently, and 2) allows us to capture variables created within ``setup``, so that constants that are not a part of any constraint can be used directly (several examples of such template models are in the new `Building Complex Models` documentation).

``Model.__init__`` calls ``setup`` with the arguments given to the constructor,  with the exception of the reserved keyword ``substitutions``. This allows for the easy creation of a named model with custom parameter values (as in the documentation's Beam example). ``setup`` methods should return an iterable (list, tuple, ConstraintSet, ...) of constraints or nothing if the model contains no constraints. To declare a submodel cost, set ``self.cost`` during ``setup``. However, we often find declaring a model's cost explicitly just before solving to be a more legible practice.

In addition to permitting us to name variables at creation, and include unconstrained variables in a model, we hope that ``setup`` methods will clarify the side effects of named model creation.

Version 0.4.2
=============
 * prototype handling of SignomialEquality constraints
 * fix an issue where solution tables printed incorrect units (despite the units being correct in the ``SolutionArray`` data structure)
 * fix ``controlpanel`` slider display for newer versions of ipywidgets
 * fix an issue where identical unit-ed variables could have different hashes
 * Make the text of several error messages more informative
 * Allow monomial approximation of monomials
 * bug fixes and improvements to TightConstraintSet
 * Don't print results table automatically (it was unwieldy for large models). To print it, ``print sol.table()``.
 * Use cvxopt's ldl kkt solver by default for more robustness to rank issues
 * Improved ``ConstraintSet.__getitem__``, only returns top-level Variable
 * Move toward the varkeys of a ConstraintSet being an immutable set
 * CPI update
 * numerous pylint fixes
 * BoundedConstraint sets added for dual feasibility debugging
 * SP sweep compatibility

Version 0.4.0
=============
 * New model for considering constraints: all constraints are considered as sets of constraints which may contain other constraints, and are asked for their substitutions / posynomial less than 1 representation as late as possible.
 * Support for calling external code during an SP solve.
 * New class KeyDict to allow referring to variables by name or with objects.
 * Many many other bug fixes, speed ups, and refactors under the hood.

Version 0.3.4
=============
 * Modular / model composition fixes and improvements
 * Working controlpanel() for Model
 * ipynb and numpy dependency fixes
 * printing fixes
 * El Capitan fix
 * slider widgets now have units

Version 0.3.2
=============
 * Assorted bug fixes
 * Assorted internal improvements and simplifications
 * Refactor signomial constraints, resulting in smarter SP heuristic
 * Simplify and strengthen equality testing for nomials
 * Not counting submodules, went from 2400 to 2500 lines of code and from 1050 to 1170 lines of docstrings and comments.

Version 0.3
===========
 * Integrated GP and SP creation under the Model class
 * Improved and simplified under-the-hood internals of GPs and SPs
 * New experimental SP heuristic
 * Improved test coverage
 * Handles vectors which are partially constants, partially free
 * Simplified interaction with Model objects and made it more pythonic
 * Added SP "step" method to allow single-stepping through an SP
 * Isolated and corrected some solver-specific behavior
 * Fully allowed substitutions of variables for 0 (commit 4631255)
 * Use "with" to create a signomials environment (commit cd8d581)
 * Continuous integration improvements, thanks @galbramc !
 * Not counting subpackages, went from 2200 to 2400 lines of code (additions were mostly longer error messages) and from 650 to 1050 lines of docstrings and comments.
 * Add automatic feasibility-analysis methods to Model and GP
 * Simplified solver logging and printing, making it easier to access solver output.

Version 0.2
===========

* Various bug fixes
* Python 3 compatibility
* Added signomial programming support (alpha quality, may be wrong)
* Added composite objectives
* Parallelized sweeping
* Better table printing
* Linked sweep variables
* Better error messages
* Closest feasible point capability
* Improved install process (no longer requires ctypesgen; auto-detects MOSEK version)
* Added examples: wind turbine, modular GP, examples from 1967 book, maintenance (part replacement)
* Documentation grew by ~70%
* Added Advanced Commands section to documentation
* Many additional unit tests (more than doubled testing lines of code)
