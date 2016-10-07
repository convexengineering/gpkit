Release Notes
*************

This page lists the changes made in each point version of gpkit.

Version 0.4.2
-------------
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
-------------
 * New model for considering constraints: all constraints are considered as sets of constraints which may contain other constraints, and are asked for their substitutions / posynomial less than 1 representation as late as possible.
 * Support for calling external code during an SP solve.
 * New class KeyDict to allow referring to variables by name or with objects.
 * Many many other bug fixes, speed ups, and refactors under the hood.

Version 0.3.2
-------------
 * Modular / model composition fixes and improvements
 * Working controlpanel() for Model
 * ipynb and numpy dependency fixes
 * printing fixes
 * El Capitan fix
 * slider widgets now have units

Version 0.3.2
-------------
 * Assorted bug fixes
 * Assorted internal improvements and simplifications
 * Refactor signomial constraints, resulting in smarter SP heuristic
 * Simplify and strengthen equality testing for nomials
 * Not counting submodules, went from 2400 to 2500 lines of code and from 1050 to 1170 lines of docstrings and comments.

Version 0.3
-----------
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
-----------

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
