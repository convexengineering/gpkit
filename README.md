# gpkit #

Python package for defining and manipulating geometric programming models, abstracting away the backend solver. Currently [MOSEK](http://mosek.com) and [CVXopt](http://cvxopt.org/) are supported.

## Overview ##

### What does gpkit look like?

Excerpted from an aircraft-design code:

```python
# Note: in Python, '**' serves as the power ('^') operator

gpkit.GP( # minimize                            # What's the lowest
         0.5*rho*S*C_D*V**2,                    # [N] TOTAL DRAG FORCE
         [ # subject to                         # that we can get, given
          Re == (rho/mu)*V*(S/A)**0.5,          # flow characteristics,
          C_f == 0.074/Re**0.2,                 # a turbulent BL approximation,
          C_D == C_D_fuse + C_D_wpar + C_D_ind  # a drag model,
          W <= 0.5*rho*S*C_L*V**2,              # flight at cruising,
          W <= 0.5*rho*S*C_Lmax*V_min**2,       # flight at takeoff,
          W == W_0 + W_w,                       # the plane's weight, and
          W_w == W_w_surf + W_w_strc, ])        # a wing weight model?
 ```

For details of that and other gpkit programs, visit our [examples folder.](http://nbviewer.ipython.org/github/appliedopt/gpkit/blob/master/gpkit/examples/simpleaircraft.ipynb)

### What can a geometric program do?

A geometric program (GP) can solve any optimization problem where [posynomials](http://en.wikipedia.org/wiki/Posynomial) form both the cost function (what you're trying to minimize or maximize, e.g. airplane fuel consumption) and the constraints (relationships that have to be true, e.g. that the plane can take off).

### Why are geometric programs useful?

It turns out [they have some nice mathematical properties](http://stanford.edu/~boyd/papers/pdf/gp_tutorial.pdf):
  - it's easy to check if something is a posynomial
  - they're quick to solve, which is good for large problems and trade-off analysis
  - solving a GP gives you an automatic sensitivity analysis (via its dual)
  - for an infeasible GP can it's easy to figure which constraints are causing the most trouble
  - a geometric program will return a globally optimal solution

Geometric programs might also have nice social properties:
  - it's clear when you can't turn an equation into a posynomial
  - many engineering equations are already posynomials
  - posynomials are simple algebraic equations, easily read

## Installation ##

1. Install Python and build dependencies:
   - Install the Python 2.7 version of [Anaconda](http://continuum.io/downloads) and then run `pip install ctypesgen`.
   - If you don't want to install Anaconda, you'll need the python packages numpy and ctypesgen, and might find sympy, scipy, and iPython Notebook to be useful as well.
   - If you want units support, install pint [`pip install pint`].
   - (Mac) If `which gcc` does not return anything, install the [Apple Command Line Tools](https://developer.apple.com/downloads/index.action?=command%20line%20tools).
2. Install either the MOSEK or CVXOPT GP solvers:
  - Download [CVXOPT](http://cvxopt.org/download/index.html):
    - (Mac/Linux) Run `python setup.py install` in the `cvxopt` folder, as noted [here](http://cvxopt.org/install/index.html#standard-installation).
    - (Windows) Follow the steps [here](http://cvxopt.org/install/index.html#building-cvxopt-for-windows).
  - Download [MOSEK](http://mosek.com/resources/downloads):
    -  (Mac OS X) Move the `mosek` folder to your home directory and follow the steps [here](http://docs.mosek.com/7.0/toolsinstall/Mac_OS_X_installation.html).
    -  (Linux) Move the `mosek` folder to your home directory and follow the steps [here](http://docs.mosek.com/7.0/toolsinstall/Linux_UNIX_installation_instructions.html).
    -  (Windows) Follow the steps [here](http://docs.mosek.com/7.0/toolsinstall/Windows_installation.html).
    - Get a MOSEK [academic license file](http://license.mosek.com/academic):
      - (Mac/Linux) Put the license file in `~/mosek/`.
      - (Windows) Put it in `Users/$USERNAME/mosek`, creating that folder if necessary.
3. Run `pip install https://github.com/appliedopt/gpkit/zipball/master` at a terminal.
  - (Windows) at an "Anaconda Command Prompt".

If you encounter any bugs during installation, email [eburn@mit.edu](mailto:eburn@mit.edu).
