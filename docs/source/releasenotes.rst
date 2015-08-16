Release Notes
*************

This page lists the changes made in each point version of gpkit.

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
